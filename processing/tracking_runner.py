"""
Thread-safe tracking runner for bbox and segmentation modes.

Adds optional annotated video output WITH motion trails in the same style/logic
as the user's known-good reference implementation:
  - Track history is maintained per raw track_id (no remap for trails).
  - For each frame, trails are drawn ONLY for tracks present in that frame.
  - Style: fixed grey polyline (default color=(230,230,230)), thickness configurable.
"""

import os
import csv
import threading
from typing import Optional, Any
from collections import defaultdict

import common
import tempfile
import shutil
import subprocess
import torch
import cv2
import time
import math
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from custom_logger import CustomLogger
from config_utils import ConfigUtils


_TLS = threading.local()
config_utils = ConfigUtils()
logger = CustomLogger(__name__)


class _AnnotatedVideoWriter:
    """Best-effort annotated video writer with OpenCV first, ffmpeg fallback."""

    def __init__(self, out_path: str, fps: float, logger_obj: Any, job_label: str = "") -> None:
        self.out_path = out_path
        self.fps = float(max(1.0, fps))
        self.logger = logger_obj
        self.job_label = job_label or "annot"
        self._w = None
        self._ff = None
        self._size = None  # (w, h)
        self._mode = None  # "opencv" | "ffmpeg" | None

    def _try_open_opencv(self, w: int, h: int) -> bool:
        candidates = ["mp4v", "avc1", "H264", "XVID", "MJPG"]
        for fourcc_str in candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                vw = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))
                if vw is not None and vw.isOpened():
                    self._w = vw
                    self._mode = "opencv"
                    self.logger.info(
                        f"[{self.job_label}] AnnotWriter: OpenCV fourcc={fourcc_str} fps={self.fps} "
                        f"size={w}x{h} out={self.out_path}"
                    )
                    return True
            except Exception:
                pass
        return False

    def _try_open_ffmpeg(self, w: int, h: int) -> bool:
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

        cmd_libx264 = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            self.out_path,
        ]

        cmd_mpeg4 = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "mpeg4",
            "-q:v", "5",
            "-pix_fmt", "yuv420p",
            self.out_path,
        ]

        for cmd in (cmd_libx264, cmd_mpeg4):
            try:
                p = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if p.stdin is None:
                    try:
                        p.kill()
                    except Exception:
                        pass
                    continue

                time.sleep(0.05)
                if p.poll() is not None:
                    continue

                self._ff = p
                self._mode = "ffmpeg"
                self.logger.info(
                    f"[{self.job_label}] AnnotWriter: ffmpeg pipe fps={self.fps} size={w}x{h} out={self.out_path}"
                )
                return True
            except Exception:
                continue

        return False

    def _ensure_open(self, frame_bgr: np.ndarray) -> None:
        if self._mode is not None:
            return
        h, w = frame_bgr.shape[:2]
        self._size = (w, h)
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

        if self._try_open_opencv(w, h):
            return
        if self._try_open_ffmpeg(w, h):
            return

        raise RuntimeError(f"[{self.job_label}] Unable to open annotated video writer for {self.out_path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8, copy=False)
        frame_bgr = np.ascontiguousarray(frame_bgr)

        self._ensure_open(frame_bgr)

        w0, h0 = self._size if self._size else (frame_bgr.shape[1], frame_bgr.shape[0])
        h, w = frame_bgr.shape[:2]
        if (w, h) != (w0, h0):
            frame_bgr = cv2.resize(frame_bgr, (w0, h0))

        if self._mode == "opencv" and self._w is not None:
            self._w.write(frame_bgr)
            return

        if self._mode == "ffmpeg" and self._ff is not None and self._ff.stdin is not None:
            try:
                self._ff.stdin.write(frame_bgr.tobytes())
            except BrokenPipeError:
                raise RuntimeError(f"[{self.job_label}] ffmpeg pipe broke while writing {self.out_path}")
            return

    def close(self) -> None:
        try:
            if self._w is not None:
                self._w.release()
        except Exception:
            pass
        self._w = None

        try:
            if self._ff is not None and self._ff.stdin is not None:
                try:
                    self._ff.stdin.close()
                except Exception:
                    pass
                try:
                    self._ff.wait(timeout=15)
                except Exception:
                    try:
                        self._ff.kill()
                    except Exception:
                        pass
        except Exception:
            pass
        self._ff = None
        self._mode = None


class TrackingRunner:
    def __init__(self) -> None:
        self.snellius_mode = bool(config_utils._safe_get_config("snellius_mode", False))
        self.segment_model = common.get_configs("segment_model")
        self.tracking_model = common.get_configs("tracking_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.seg_tracker = common.get_configs("seg_tracker")

        try:
            self.track_buffer_sec = float(common.get_configs("track_buffer_sec"))
        except Exception:
            self.track_buffer_sec = 2.0

        try:
            self.confidence = float(common.get_configs("confidence") or 0.0)
        except Exception:
            self.confidence = 0.0

    def _reset_ultralytics_tracker(self, model) -> None:
        try:
            if model is None:
                return
            pred = getattr(model, "predictor", None)
            try:
                ds = getattr(pred, "dataset", None)
                cap = getattr(ds, "cap", None)
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                model.predictor = None
            except Exception:
                pass
        except Exception:
            pass

    def get_thread_models(self, bbox_mode: bool, seg_mode: bool) -> tuple[Optional[YOLO], Optional[YOLO]]:
        if bbox_mode:
            if not hasattr(_TLS, "bbox_model") or _TLS.bbox_model is None:
                _TLS.bbox_model = YOLO(self.tracking_model)
        else:
            _TLS.bbox_model = None

        if seg_mode:
            if not hasattr(_TLS, "seg_model") or _TLS.seg_model is None:
                _TLS.seg_model = YOLO(self.segment_model)
        else:
            _TLS.seg_model = None

        return _TLS.bbox_model, _TLS.seg_model

    def make_tracker_config(self, tracker_path: str, video_fps: int) -> str:
        if not (isinstance(tracker_path, str) and tracker_path.endswith(".yaml") and os.path.isfile(tracker_path)):
            return tracker_path

        base = os.path.basename(tracker_path)
        if base not in ("bbox_custom_tracker.yaml", "seg_custom_tracker.yaml"):
            return tracker_path

        try:
            import yaml
        except Exception:
            return tracker_path

        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return tracker_path

        cfg["track_buffer"] = float(self.track_buffer_sec) * float(video_fps)

        tmp_dir = tempfile.mkdtemp(prefix="tracker_cfg_")
        tmp_path = os.path.join(tmp_dir, base)
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        return tmp_path

    def tracking_mode_threadsafe(
        self,
        input_video_path: str,
        video_fps: int,
        bbox_mode: bool,
        seg_mode: bool,
        bbox_csv_out: Optional[str] = None,
        seg_csv_out: Optional[str] = None,
        annotated_video_out: Optional[str] = None,
        bbox_model: YOLO | None = None,
        seg_model: YOLO | None = None,
        flush_every_n_frames: int = 3000,
        job_label: str = "",
        show_frame_pbar: bool = True,
        tqdm_position: int = 1,
        postfix_every_n: int = 30,
        remap_track_ids_per_segment: bool = True,
    ) -> None:
        if bbox_mode and bbox_csv_out is None:
            raise ValueError("bbox_mode=True requires bbox_csv_out")
        if seg_mode and seg_csv_out is None:
            raise ValueError("seg_mode=True requires seg_csv_out")

        flush_every_n_frames = int(config_utils._safe_get_config("flush_every_n_frames", flush_every_n_frames))
        progress_log_every_sec = float(config_utils._safe_get_config("progress_log_every_sec", 5.0))

        if bbox_model is None or (seg_mode and seg_model is None):
            bbox_m, seg_m = self.get_thread_models(bbox_mode, seg_mode)
            if bbox_model is None:
                bbox_model = bbox_m
            if seg_model is None:
                seg_model = seg_m

        if bbox_mode and bbox_model is None:
            raise ValueError("bbox_mode=True but bbox_model is None")
        if seg_mode and seg_model is None:
            raise ValueError("seg_mode=True but seg_model is None")

        device: Any = 0 if torch.cuda.is_available() else "cpu"

        bbox_tracker_eff = self.make_tracker_config(self.bbox_tracker, video_fps) if bbox_mode else self.bbox_tracker
        seg_tracker_eff = self.make_tracker_config(self.seg_tracker, video_fps) if seg_mode else self.seg_tracker

        bbox_header = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]
        seg_header = ["yolo-id", "mask-polygon", "unique-id", "confidence", "frame-count"]

        bbox_buf: list[list] = []
        seg_buf: list[list] = []
        frame_count = 0

        bbox_id_map: dict[int, int] = {}
        seg_id_map: dict[int, int] = {}
        bbox_next = [1]
        seg_next = [1]

        # --- Trail settings to match your old behavior ---
        # Fixed length history in frames (default 30 as in your code)
        trail_len = int(config_utils._safe_get_config("trail_len", 30))
        trail_len = max(2, trail_len)

        # Fixed grey and thick line (default matches your (230,230,230) and LINE_THICKNESS*5 feel)
        trail_color = tuple(config_utils._safe_get_config("trail_color_bgr", (230, 230, 230)))  # type: ignore
        trail_thickness = int(config_utils._safe_get_config("trail_thickness", 10))

        # Separate histories for seg and bbox, keyed by RAW track_id (as in your code)
        seg_track_history = defaultdict(list)   # track_id -> [(x,y), ...]
        bbox_track_history = defaultdict(list)  # track_id -> [(x,y), ...]

        def _map_id(raw_id: int, id_map: dict[int, int], next_id_holder: list[int]) -> int:
            if raw_id is None or raw_id < 0:
                return -1
            if raw_id in id_map:
                return id_map[raw_id]
            nid = next_id_holder[0]
            id_map[raw_id] = nid
            next_id_holder[0] += 1
            return nid

        def _is_linalg_error(e: BaseException) -> bool:
            return isinstance(e, np.linalg.LinAlgError) or (e.__class__.__name__ == "LinAlgError")

        # progress bar total
        total_frames: Optional[int] = None
        if show_frame_pbar:
            try:
                cap0 = cv2.VideoCapture(input_video_path)
                if cap0.isOpened():
                    tf = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    total_frames = tf if tf > 0 else None
                cap0.release()
            except Exception:
                total_frames = None

        pbar = None
        if show_frame_pbar:
            desc = f"frames: {job_label}" if job_label else "frames"
            pbar = tqdm(
                total=total_frames,
                desc=desc,
                unit="f",
                dynamic_ncols=True,
                position=tqdm_position,
                leave=False,
                mininterval=1.0,
            )

        bbox_f = seg_f = None
        bbox_w = seg_w = None

        annot_writer: Optional[_AnnotatedVideoWriter] = None
        if annotated_video_out:
            os.makedirs(os.path.dirname(annotated_video_out) or ".", exist_ok=True)
            annot_writer = _AnnotatedVideoWriter(annotated_video_out, float(video_fps), logger, job_label=job_label)

        def _flush_buffers() -> None:
            nonlocal bbox_buf, seg_buf
            if bbox_w and bbox_buf:
                bbox_w.writerows(bbox_buf)
                bbox_buf.clear()
                if bbox_f:
                    bbox_f.flush()
            if seg_w and seg_buf:
                seg_w.writerows(seg_buf)
                seg_buf.clear()
                if seg_f:
                    seg_f.flush()

        def _log_progress_periodic(last_log_t: float) -> float:
            now = time.time()
            if progress_log_every_sec > 0 and (now - last_log_t) >= progress_log_every_sec:
                logger.info(f"[{job_label}] progress: processed_frames={frame_count}")
                return now
            return last_log_t

        def _draw_trails_like_reference(
            annotated_frame: np.ndarray,
            boxes_xywh_px: Optional[np.ndarray],
            track_ids: list[int],
            history: dict,
        ) -> None:
            """Reproduces your old trail code: update + draw only for tracks present in this frame."""
            if annotated_frame is None or boxes_xywh_px is None:
                return
            if boxes_xywh_px.size == 0 or not track_ids:
                return

            # Ensure shape (N,4)
            if len(boxes_xywh_px.shape) != 2 or boxes_xywh_px.shape[1] < 2:
                return

            # Iterate per detection in THIS frame only
            for box, tid in zip(boxes_xywh_px, track_ids):
                try:
                    x = float(box[0])  # center x in pixels
                    y = float(box[1])  # center y in pixels
                    track = history[tid]
                    track.append((x, y))
                    if len(track) > trail_len:
                        track.pop(0)

                    # Same approach as your code
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=trail_color,           # (230,230,230)
                        thickness=trail_thickness,   # e.g., LINE_THICKNESS*5
                        lineType=cv2.LINE_AA,
                    )
                except Exception:
                    pass

        def _annotate_write_frame_with_trails(
            r: Any,
            mode: str,
            fallback_frame: Optional[np.ndarray] = None,
        ) -> None:
            """Write annotated MP4 frame, adding trails in the same style as the reference."""
            if annot_writer is None:
                return

            # base annotated image
            img = None
            if r is not None:
                try:
                    img = r.plot()
                except Exception:
                    img = None
            if img is None and fallback_frame is not None:
                try:
                    img = fallback_frame.copy()
                except Exception:
                    img = None
            if img is None:
                return

            # Extract current-frame xywh (pixel) + raw track ids
            try:
                boxes = getattr(r, "boxes", None) if r is not None else None
                if boxes is not None and getattr(boxes, "xywh", None) is not None and getattr(boxes, "id", None) is not None:
                    xywh_px = boxes.xywh.detach().cpu().numpy()  # pixel xywh
                    raw_ids = boxes.id.int().detach().cpu().tolist()
                else:
                    xywh_px = None
                    raw_ids = []
            except Exception:
                xywh_px = None
                raw_ids = []

            # Draw trails only for tracks detected in this frame, like your code
            if mode == "seg":
                _draw_trails_like_reference(img, xywh_px, raw_ids, seg_track_history)
            else:
                _draw_trails_like_reference(img, xywh_px, raw_ids, bbox_track_history)

            try:
                annot_writer.write(img)
            except Exception as e:
                logger.warning(f"[{job_label}] annotated write failed at frame={frame_count}: {e}")

        def _opencv_loop_from_current_frame() -> None:
            nonlocal frame_count, bbox_buf, seg_buf
            last_log_t = time.time()

            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {input_video_path}")

            if frame_count > 0:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_count))
                except Exception:
                    pass

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame_count += 1
                    persist_flag = frame_count > 1

                    # Prefer seg annotated output if available; otherwise bbox.
                    annot_r = None
                    annot_mode = "bbox"

                    # --- SEG ---
                    if seg_mode:
                        seg_results = None
                        try:
                            seg_results = seg_model.track(  # type: ignore
                                frame,
                                tracker=seg_tracker_eff,
                                persist=persist_flag,
                                conf=self.confidence,
                                verbose=False,
                                device=device,
                                save=False,
                                save_txt=False,
                                show=False,
                            )
                        except Exception as e:
                            msg = "LinAlgError" if _is_linalg_error(e) else "error"
                            logger.warning(f"[{job_label}][Frame {frame_count}] SEG {msg}; reset+retry. err={e}")
                            self._reset_ultralytics_tracker(seg_model)  # type: ignore
                            try:
                                seg_results = seg_model.track(  # type: ignore
                                    frame,
                                    tracker=seg_tracker_eff,
                                    persist=False,
                                    conf=self.confidence,
                                    verbose=False,
                                    device=device,
                                    save=False,
                                    save_txt=False,
                                    show=False,
                                )
                            except Exception as e2:
                                logger.warning(f"[{job_label}][Frame {frame_count}] SEG retry failed; skipping seg parse. err={e2}")
                                seg_results = None

                        if seg_results is not None:
                            r = seg_results[0]
                            annot_r = r
                            annot_mode = "seg"

                            boxes = getattr(r, "boxes", None)
                            masks = getattr(r, "masks", None)

                            if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:
                                n = int(boxes.xywhn.size(0))
                                cls_list = boxes.cls.int().cpu().tolist()
                                raw_id_list = (
                                    boxes.id.int().cpu().tolist()
                                    if getattr(boxes, "id", None) is not None
                                    else [-1] * n
                                )
                                conf_list = (
                                    boxes.conf.cpu().tolist()
                                    if getattr(boxes, "conf", None) is not None
                                    else [math.nan] * n
                                )

                                if masks is not None and getattr(masks, "xyn", None) is not None:
                                    polys = masks.xyn
                                    m = min(len(polys), n)
                                    for i in range(m):
                                        poly = polys[i]
                                        flat = []
                                        for x, y in poly:
                                            flat.append(str(float(x)))
                                            flat.append(str(float(y)))

                                        raw_tid = int(raw_id_list[i]) if i < len(raw_id_list) else -1
                                        tid = _map_id(raw_tid, seg_id_map, seg_next) if remap_track_ids_per_segment else raw_tid
                                        seg_buf.append([int(cls_list[i]), " ".join(flat), int(tid), float(conf_list[i]), frame_count])

                    # --- BBOX ---
                    if bbox_mode:
                        bbox_results = None
                        try:
                            bbox_results = bbox_model.track(  # type: ignore
                                frame,
                                tracker=bbox_tracker_eff,
                                persist=persist_flag,
                                conf=self.confidence,
                                verbose=False,
                                device=device,
                                save=False,
                                save_txt=False,
                                show=False,
                            )
                        except Exception as e:
                            msg = "LinAlgError" if _is_linalg_error(e) else "error"
                            logger.warning(f"[{job_label}][Frame {frame_count}] BBOX {msg}; reset+retry. err={e}")
                            self._reset_ultralytics_tracker(bbox_model)  # type: ignore
                            try:
                                bbox_results = bbox_model.track(  # type: ignore
                                    frame,
                                    tracker=bbox_tracker_eff,
                                    persist=False,
                                    conf=self.confidence,
                                    verbose=False,
                                    device=device,
                                    save=False,
                                    save_txt=False,
                                    show=False,
                                )
                            except Exception as e2:
                                logger.warning(f"[{job_label}][Frame {frame_count}] BBOX retry failed; skipping bbox parse. err={e2}")
                                bbox_results = None

                        if bbox_results is not None:
                            r = bbox_results[0]
                            if annot_r is None:
                                annot_r = r
                                annot_mode = "bbox"

                            boxes = getattr(r, "boxes", None)
                            if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:
                                xywhn = boxes.xywhn.cpu().tolist()
                                cls_list = boxes.cls.int().cpu().tolist()
                                raw_id_list = (
                                    boxes.id.int().cpu().tolist()
                                    if getattr(boxes, "id", None) is not None
                                    else [-1] * len(xywhn)
                                )
                                conf_list = (
                                    boxes.conf.cpu().tolist()
                                    if getattr(boxes, "conf", None) is not None
                                    else [math.nan] * len(xywhn)
                                )

                                for (x, y, w, h), c, raw_tid, confv in zip(xywhn, cls_list, raw_id_list, conf_list):
                                    raw_tid = int(raw_tid) if raw_tid is not None else -1
                                    tid = _map_id(raw_tid, bbox_id_map, bbox_next) if remap_track_ids_per_segment else raw_tid
                                    bbox_buf.append([int(c), float(x), float(y), float(w), float(h), int(tid), float(confv), frame_count])

                    # Annotated MP4: draw trails like reference and write
                    if annot_r is not None:
                        _annotate_write_frame_with_trails(annot_r, mode=annot_mode, fallback_frame=frame)
                    else:
                        # No detections; write original frame if annotation requested
                        if annot_writer is not None:
                            annot_writer.write(frame)

                    if frame_count % flush_every_n_frames == 0:
                        _flush_buffers()

                    if pbar is not None:
                        pbar.update(1)
                        if postfix_every_n and (frame_count % postfix_every_n == 0):
                            pbar.set_postfix({
                                "f": frame_count,
                                "bbox_buf": (0 if not bbox_mode else len(bbox_buf)),
                                "seg_buf": (0 if not seg_mode else len(seg_buf)),
                            })

                    last_log_t = _log_progress_periodic(last_log_t)

                _flush_buffers()

            finally:
                cap.release()

        try:
            # Open CSV outputs
            if bbox_mode and bbox_csv_out:
                os.makedirs(os.path.dirname(bbox_csv_out) or ".", exist_ok=True)
                bbox_new = not os.path.exists(bbox_csv_out)
                bbox_f = open(bbox_csv_out, "a", newline="", encoding="utf-8")
                bbox_w = csv.writer(bbox_f)
                if bbox_new:
                    bbox_w.writerow(bbox_header)

            if seg_mode and seg_csv_out:
                os.makedirs(os.path.dirname(seg_csv_out) or ".", exist_ok=True)
                seg_new = not os.path.exists(seg_csv_out)
                seg_f = open(seg_csv_out, "a", newline="", encoding="utf-8")
                seg_w = csv.writer(seg_f)
                if seg_new:
                    seg_w.writerow(seg_header)

            try:
                # Snellius streaming (single-mode only); otherwise OpenCV loop
                if self.snellius_mode:
                    if bbox_mode and seg_mode:
                        logger.info(f"[{job_label}] Snellius: bbox+seg enabled -> OpenCV per-frame loop.")
                        _opencv_loop_from_current_frame()
                        return

                    last_log_t = time.time()

                    def _iter_results(it):
                        for rr in it:
                            if isinstance(rr, (list, tuple)):
                                yield (rr[0] if rr else None)
                            else:
                                yield rr

                    try:
                        if bbox_mode:
                            bbox_iter = bbox_model.track(  # type: ignore
                                source=input_video_path,
                                stream=True,
                                persist=True,
                                tracker=bbox_tracker_eff,
                                conf=self.confidence,
                                verbose=False,
                                device=device,
                                save=False,
                                save_txt=False,
                                show=False,
                            )

                            for r in _iter_results(bbox_iter):
                                if r is None:
                                    continue
                                frame_count += 1

                                boxes = getattr(r, "boxes", None)
                                if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:
                                    xywhn = boxes.xywhn.cpu().tolist()
                                    cls_list = boxes.cls.int().cpu().tolist()
                                    raw_id_list = (
                                        boxes.id.int().cpu().tolist()
                                        if getattr(boxes, "id", None) is not None
                                        else [-1] * len(xywhn)
                                    )
                                    conf_list = (
                                        boxes.conf.cpu().tolist()
                                        if getattr(boxes, "conf", None) is not None
                                        else [math.nan] * len(xywhn)
                                    )

                                    for (x, y, w, h), c, raw_tid, confv in zip(xywhn, cls_list, raw_id_list, conf_list):
                                        raw_tid = int(raw_tid) if raw_tid is not None else -1
                                        tid = _map_id(raw_tid, bbox_id_map, bbox_next) if remap_track_ids_per_segment else raw_tid
                                        bbox_buf.append([int(c), float(x), float(y), float(w), float(h), int(tid), float(confv), frame_count])

                                # write annotated with trails
                                _annotate_write_frame_with_trails(r, mode="bbox", fallback_frame=None)

                                if frame_count % flush_every_n_frames == 0:
                                    _flush_buffers()

                                if pbar is not None:
                                    pbar.update(1)
                                    if postfix_every_n and (frame_count % postfix_every_n == 0):
                                        pbar.set_postfix({"f": frame_count, "bbox_buf": len(bbox_buf)})

                                last_log_t = _log_progress_periodic(last_log_t)

                            _flush_buffers()
                            return

                        if seg_mode:
                            seg_iter = seg_model.track(  # type: ignore
                                source=input_video_path,
                                stream=True,
                                persist=True,
                                tracker=seg_tracker_eff,
                                conf=self.confidence,
                                verbose=False,
                                device=device,
                                save=False,
                                save_txt=False,
                                show=False,
                            )

                            for r in _iter_results(seg_iter):
                                if r is None:
                                    continue
                                frame_count += 1

                                boxes = getattr(r, "boxes", None)
                                masks = getattr(r, "masks", None)

                                if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:
                                    n = int(boxes.xywhn.size(0))
                                    cls_list = boxes.cls.int().cpu().tolist()
                                    raw_id_list = (
                                        boxes.id.int().cpu().tolist()
                                        if getattr(boxes, "id", None) is not None
                                        else [-1] * n
                                    )
                                    conf_list = (
                                        boxes.conf.cpu().tolist()
                                        if getattr(boxes, "conf", None) is not None
                                        else [math.nan] * n
                                    )

                                    if masks is not None and getattr(masks, "xyn", None) is not None:
                                        polys = masks.xyn
                                        m = min(len(polys), n)
                                        for i in range(m):
                                            poly = polys[i]
                                            flat = []
                                            for x, y in poly:
                                                flat.append(str(float(x)))
                                                flat.append(str(float(y)))

                                            raw_tid = int(raw_id_list[i]) if i < len(raw_id_list) else -1
                                            tid = _map_id(raw_tid, seg_id_map, seg_next) if remap_track_ids_per_segment else raw_tid
                                            seg_buf.append([int(cls_list[i]), " ".join(flat), int(tid), float(conf_list[i]), frame_count])

                                # write annotated with trails
                                _annotate_write_frame_with_trails(r, mode="seg", fallback_frame=None)

                                if frame_count % flush_every_n_frames == 0:
                                    _flush_buffers()

                                if pbar is not None:
                                    pbar.update(1)
                                    if postfix_every_n and (frame_count % postfix_every_n == 0):
                                        pbar.set_postfix({"f": frame_count, "seg_buf": len(seg_buf)})

                                last_log_t = _log_progress_periodic(last_log_t)

                            _flush_buffers()
                            return

                        return

                    except Exception as e:
                        _flush_buffers()
                        msg = "LinAlgError" if _is_linalg_error(e) else "error"
                        logger.warning(
                            f"[{job_label}] Snellius streaming crashed with {msg} at processed_frames={frame_count}. "
                            f"Reset + fallback to OpenCV. err={e}"
                        )

                        try:
                            if bbox_mode and bbox_model is not None:
                                self._reset_ultralytics_tracker(bbox_model)
                        except Exception:
                            pass
                        try:
                            if seg_mode and seg_model is not None:
                                self._reset_ultralytics_tracker(seg_model)
                        except Exception:
                            pass

                        _opencv_loop_from_current_frame()
                        return

                _opencv_loop_from_current_frame()
                return

            finally:
                if pbar is not None:
                    pbar.close()

                for p in (bbox_tracker_eff, seg_tracker_eff):
                    if isinstance(p, str) and "tracker_cfg_" in p:
                        try:
                            shutil.rmtree(os.path.dirname(p), ignore_errors=True)
                        except Exception:
                            pass

        finally:
            try:
                if bbox_f:
                    bbox_f.close()
            except Exception:
                pass
            try:
                if seg_f:
                    seg_f.close()
            except Exception:
                pass
            try:
                if annot_writer is not None:
                    annot_writer.close()
            except Exception:
                pass
