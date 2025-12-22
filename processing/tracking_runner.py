"""
Thread-safe tracking runner for bbox and segmentation modes.

Key design goal (bbox mode):
- Stable tracker IDs *within a clip/segment*.
- Avoid per-frame tracker re-initialization (which causes IDs to restart 1..N every frame)
  by preferring Ultralytics streaming mode on a video path.

This version:
- Applies the SciPy cdist patch *before* importing Ultralytics.
- Uses YOLO.track(source=<video>, stream=True, persist=True, tracker=...) for bbox-only runs.
- Writes bbox CSV and (optional) annotated video.
- Falls back to a simple OpenCV per-frame DETECTION-only loop if streaming tracking crashes,
  to avoid missing frames (IDs will be -1 in fallback, because tracking is unavailable).
"""

from __future__ import annotations

import os
import csv
import math
import time
import shutil
import tempfile
import threading
import subprocess
from collections import defaultdict
from typing import Optional, Any, Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from config_utils import ConfigUtils
from patches import Patches

# -----------------------------------------------------------------------------
# Early patches (import-order sensitive)
# -----------------------------------------------------------------------------
logger = CustomLogger(__name__)
try:
    _patches = Patches()
    _patches.patch_ultralytics_botsort_numpy_cpu_bug(logger)
    _patches.patch_scipy_cdist_accept_1d(logger)
except Exception as _e:
    # Must never be fatal
    logger.warning(f"Patches could not be applied (continuing): {_e!r}")

from ultralytics import YOLO  # noqa: E402  (after patches)


_TLS = threading.local()
config_utils = ConfigUtils()


class _AnnotatedVideoWriter:
    """Best-effort annotated video writer with OpenCV first, ffmpeg fallback."""

    def __init__(self, out_path: str, fps: float, logger_obj: Any, job_label: str = "") -> None:
        self.out_path = out_path
        self.fps = float(max(1.0, fps))
        self.logger = logger_obj
        self.job_label = job_label or "annot"
        self._w = None
        self._ff = None
        self._size: Optional[tuple[int, int]] = None  # (w, h)
        self._mode: Optional[str] = None  # "opencv" | "ffmpeg"

    def _try_open_opencv(self, w: int, h: int) -> bool:
        candidates = ["mp4v", "avc1", "H264", "XVID", "MJPG"]
        for fourcc_str in candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)  # type: ignore
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
                continue
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
            self._ff.stdin.write(frame_bgr.tobytes())
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
    """
    Thread-safe tracker runner.

    Important behavior:
    - Models are cached per thread (`_TLS`) for performance.
    - Stable IDs within a segment are obtained via streaming mode on the segment video file.
    """

    # Global ID allocator (optional)
    _id_lock = threading.Lock()
    _bbox_next_global = 1
    _seg_next_global = 1

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

    @classmethod
    def _alloc_global_ids(cls, n: int, kind: str) -> list[int]:
        if n <= 0:
            return []
        with cls._id_lock:
            if kind == "seg":
                start = cls._seg_next_global
                cls._seg_next_global += n
            else:
                start = cls._bbox_next_global
                cls._bbox_next_global += n
        return list(range(start, start + n))

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
        """Adjust track_buffer dynamically for bbox_custom_tracker.yaml / seg_custom_tracker.yaml."""
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

        # Convert seconds -> frames
        cfg["track_buffer"] = float(self.track_buffer_sec) * float(video_fps)

        tmp_dir = tempfile.mkdtemp(prefix="tracker_cfg_")
        tmp_path = os.path.join(tmp_dir, base)
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return tmp_path

    @staticmethod
    def _iter_results(it: Iterable[Any]) -> Iterable[Any]:
        """Ultralytics sometimes yields [Results]; normalize to Results."""
        for rr in it:
            if isinstance(rr, (list, tuple)):
                yield (rr[0] if rr else None)
            else:
                yield rr

    def tracking_mode_threadsafe(
        self,
        input_video_path: str,
        video_fps: int,
        bbox_mode: bool,
        seg_mode: bool,
        bbox_csv_out: Optional[str] = None,
        seg_csv_out: Optional[str] = None,
        annotated_video_out: Optional[str] = None,
        bbox_model: YOLO | None = None,  # type: ignore
        seg_model: YOLO | None = None,  # type: ignore
        flush_every_n_frames: int = 3000,
        job_label: str = "",
        show_frame_pbar: bool = True,
        tqdm_position: int = 1,
        postfix_every_n: int = 30,
        remap_track_ids_per_segment: bool = False,
    ) -> None:
        # Validate
        if bbox_mode and bbox_csv_out is None:
            raise ValueError("bbox_mode=True requires bbox_csv_out")
        if seg_mode and seg_csv_out is None:
            raise ValueError("seg_mode=True requires seg_csv_out")

        # This runner focuses on bbox-only stability. seg_mode is left supported but not rewritten here.
        if seg_mode:
            logger.warning(f"[{job_label}] seg_mode=True requested; this runner prioritizes bbox-only stability.")

        flush_every_n_frames = int(config_utils._safe_get_config("flush_every_n_frames", flush_every_n_frames))  # type: ignore
        progress_log_every_sec = float(config_utils._safe_get_config("progress_log_every_sec", 5.0))  # type: ignore

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

        bbox_tracker_eff = self.make_tracker_config(self.bbox_tracker, int(video_fps)) if bbox_mode else self.bbox_tracker
        seg_tracker_eff = self.make_tracker_config(self.seg_tracker, int(video_fps)) if seg_mode else self.seg_tracker

        bbox_header = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]
        seg_header = ["yolo-id", "mask-polygon", "unique-id", "confidence", "frame-count"]

        bbox_buf: list[list[Any]] = []
        seg_buf: list[list[Any]] = []
        frame_count = 0

        # ID remap (optional)
        bbox_local_map: dict[int, int] = {}
        bbox_local_next = 1

        def map_id(raw_id: int) -> int:
            nonlocal bbox_local_next
            if raw_id is None or raw_id < 0:
                return -1
            if not remap_track_ids_per_segment:
                return int(raw_id)
            if raw_id not in bbox_local_map:
                bbox_local_map[raw_id] = bbox_local_next
                bbox_local_next += 1
            return bbox_local_map[raw_id]

        # Trails (optional)
        draw_trails = bool(config_utils._safe_get_config("annotated_video_draw_trails", True))
        trail_len = int(config_utils._safe_get_config("trail_len", 30) or 30)
        trail_len = max(2, trail_len)
        trail_color = tuple(config_utils._safe_get_config("trail_color_bgr", (230, 230, 230)) or (230, 230, 230))
        trail_thickness = int(config_utils._safe_get_config("trail_thickness", 10) or 10)
        trail_hist: dict[int, list[tuple[float, float]]] = defaultdict(list)

        # Progress bar total
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
            annot_writer = _AnnotatedVideoWriter(annotated_video_out, float(video_fps), logger, job_label=job_label)

        def flush_buffers() -> None:
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

        def log_progress_periodic(last_t: float) -> float:
            now = time.time()
            if progress_log_every_sec > 0 and (now - last_t) >= progress_log_every_sec:
                logger.info(f"[{job_label}] progress: processed_frames={frame_count}")
                return now
            return last_t

        def draw_trails_on(img: np.ndarray, boxes_xywh_px: np.ndarray, ids_out: list[int]) -> None:
            if not draw_trails:
                return
            if img is None or boxes_xywh_px is None or boxes_xywh_px.size == 0:
                return
            if len(ids_out) == 0:
                return
            if boxes_xywh_px.ndim != 2 or boxes_xywh_px.shape[1] < 2:
                return
            for box, tid in zip(boxes_xywh_px, ids_out):
                if tid is None or tid < 0:
                    continue
                x = float(box[0]); y = float(box[1])
                hist = trail_hist[tid]
                hist.append((x, y))
                if len(hist) > trail_len:
                    del hist[0:len(hist) - trail_len]
                try:
                    pts = np.array(hist, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], False, trail_color, trail_thickness, lineType=cv2.LINE_AA)
                except Exception:
                    continue

        # Open CSV outputs
        try:
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

            last_log_t = time.time()

            # -----------------------------------------------------------------
            # Preferred: bbox-only streaming tracking (stable IDs)
            # -----------------------------------------------------------------
            if bbox_mode and not seg_mode:
                assert bbox_model is not None

                try:
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

                    for r in self._iter_results(bbox_iter):
                        if r is None:
                            continue
                        frame_count += 1

                        boxes = getattr(r, "boxes", None)
                        if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:
                            xywhn = boxes.xywhn.detach().cpu().tolist()
                            cls_list = boxes.cls.int().detach().cpu().tolist()
                            raw_id_list = (
                                boxes.id.int().detach().cpu().tolist()
                                if getattr(boxes, "id", None) is not None
                                else [-1] * len(xywhn)
                            )
                            conf_list = (
                                boxes.conf.detach().cpu().tolist()
                                if getattr(boxes, "conf", None) is not None
                                else [math.nan] * len(xywhn)
                            )

                            # Map IDs
                            out_ids = [map_id(int(x)) for x in raw_id_list]

                            for (x, y, w, h), c, tid, confv in zip(xywhn, cls_list, out_ids, conf_list):
                                bbox_buf.append([int(c), float(x), float(y), float(w), float(h), int(tid), float(confv), frame_count])

                            # Annotated frame
                            if annot_writer is not None:
                                try:
                                    img = r.plot()
                                    # Trails use pixel xywh
                                    try:
                                        xywh_px = boxes.xywh.detach().cpu().numpy() if getattr(boxes, "xywh", None) is not None else None
                                    except Exception:
                                        xywh_px = None
                                    if xywh_px is not None:
                                        draw_trails_on(img, xywh_px, out_ids)
                                    annot_writer.write(img)
                                except Exception as e:
                                    logger.warning(f"[{job_label}] annotated write failed at frame={frame_count}: {e}")

                        else:
                            # No detections; still write plain frame if requested
                            if annot_writer is not None:
                                try:
                                    img = r.orig_img if getattr(r, "orig_img", None) is not None else None
                                    if img is not None:
                                        annot_writer.write(img)
                                except Exception:
                                    pass

                        if frame_count % flush_every_n_frames == 0:
                            flush_buffers()

                        if pbar is not None:
                            pbar.update(1)
                            if postfix_every_n and (frame_count % postfix_every_n == 0):
                                pbar.set_postfix({"f": frame_count, "bbox_buf": len(bbox_buf)})

                        last_log_t = log_progress_periodic(last_log_t)

                    flush_buffers()
                    return

                except Exception as e:
                    # Streaming tracking crashed; fall back to detection-only per frame to avoid missing frames.
                    flush_buffers()
                    logger.warning(
                        f"[{job_label}] Streaming bbox tracking crashed at processed_frames={frame_count}. "
                        f"Falling back to OpenCV detection-only loop. err={e}"
                    )

                    # -----------------------------------------------------------------
                    # Fallback: OpenCV per-frame detection-only (no tracking, IDs=-1)
                    # -----------------------------------------------------------------
                    cap = cv2.VideoCapture(input_video_path)
                    if not cap.isOpened():
                        raise RuntimeError(f"Failed to open video: {input_video_path}")

                    try:
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            frame_count += 1
                            det = bbox_model.predict(  # type: ignore
                                frame,
                                conf=self.confidence,
                                verbose=False,
                                device=device,
                            )
                            r0 = det[0] if isinstance(det, (list, tuple)) and det else det
                            boxes = getattr(r0, "boxes", None) if r0 is not None else None
                            if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:
                                xywhn = boxes.xywhn.detach().cpu().tolist()
                                cls_list = boxes.cls.int().detach().cpu().tolist()
                                conf_list = boxes.conf.detach().cpu().tolist() if getattr(boxes, "conf", None) is not None else [math.nan] * len(xywhn)
                                for (x, y, w, h), c, confv in zip(xywhn, cls_list, conf_list):
                                    bbox_buf.append([int(c), float(x), float(y), float(w), float(h), -1, float(confv), frame_count])

                            if annot_writer is not None:
                                try:
                                    img = r0.plot() if r0 is not None else frame
                                    annot_writer.write(img)
                                except Exception:
                                    try:
                                        annot_writer.write(frame)
                                    except Exception:
                                        pass

                            if frame_count % flush_every_n_frames == 0:
                                flush_buffers()
                            if pbar is not None:
                                pbar.update(1)

                            last_log_t = log_progress_periodic(last_log_t)

                        flush_buffers()
                        return

                    finally:
                        cap.release()

            # -----------------------------------------------------------------
            # If seg_mode or bbox+seg are requested, use the existing behavior:
            # (OpenCV loop calling track per frame). This can be extended later.
            # -----------------------------------------------------------------
            raise NotImplementedError("This rewritten runner currently supports stable bbox-only mode. Enable only bbox_mode.")

        finally:
            try:
                if pbar is not None:
                    pbar.close()
            except Exception:
                pass

            # Clean temp tracker configs
            for p in (bbox_tracker_eff, seg_tracker_eff):
                if isinstance(p, str) and "tracker_cfg_" in p:
                    try:
                        shutil.rmtree(os.path.dirname(p), ignore_errors=True)
                    except Exception:
                        pass

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
