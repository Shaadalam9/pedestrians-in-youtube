import os
import csv
import threading
from typing import Optional, Any
import common
import tempfile
import shutil
import torch
import cv2
import time
import math
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from custom_logger import CustomLogger
from config_utils import ConfigUtils


# Thread-local storage for per-thread model instances
_TLS = threading.local()
config_utils = ConfigUtils()
logger = CustomLogger(__name__)


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

        # Tracking behavior
        try:
            self.confidence = float(common.get_configs("confidence") or 0.0)
        except Exception:
            self.confidence = 0.0

    def _reset_ultralytics_tracker(self, model) -> None:
        """
        Hard reset for Ultralytics tracking/predictor state.

        Important: do NOT set predictor.trackers = None.
        Some Ultralytics versions assume predictor.trackers is indexable.
        The safest cross-version reset is to drop the predictor entirely so it is rebuilt cleanly.
        """
        try:
            if model is None:
                return

            pred = getattr(model, "predictor", None)

            # Best-effort: release any internal VideoCapture held by predictor/dataset (varies by version)
            try:
                ds = getattr(pred, "dataset", None)
                cap = getattr(ds, "cap", None)
                if cap is not None:
                    cap.release()
            except Exception:
                pass

            # Force a full rebuild on next .track()/.predict()
            try:
                model.predictor = None
            except Exception:
                pass

        except Exception:
            # Never allow reset itself to crash the pipeline
            pass

    def _write_rows_csv(self, path: str, header: list[str], rows: list[list]) -> None:
        if not rows:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerows(rows)

    def get_thread_models(self, bbox_mode: bool, seg_mode: bool) -> tuple[Optional[YOLO], Optional[YOLO]]:
        """
        Thread-local cache: each worker thread gets its own YOLO model instances.
        This avoids race conditions inside Ultralytics trackers.
        """
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
        """
        Thread-safe alternative to in-place YAML edits:
        - Only for bbox_custom_tracker.yaml / seg_custom_tracker.yaml
        - Creates a temp copy with track_buffer = track_buffer_sec * fps
        """
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

    def tracking_mode_threadsafe(self, input_video_path: str, video_fps: int, bbox_mode: bool, seg_mode: bool,
                                 bbox_csv_out: Optional[str] = None, seg_csv_out: Optional[str] = None,
                                 bbox_model: YOLO | None = None, seg_model: YOLO | None = None,
                                 flush_every_n_frames: int = 3000, job_label: str = "", show_frame_pbar: bool = True,
                                 tqdm_position: int = 1, postfix_every_n: int = 30,
                                 remap_track_ids_per_segment: bool = True) -> None:
        """
        Thread-safe tracking:
          - writes CSVs directly (same schema)
          - Snellius mode: tries Ultralytics streaming (fast)
          - If streaming crashes (LinAlgError etc.), falls back to OpenCV per-frame loop
          - OpenCV loop includes robust per-frame exception handling:
              * on tracker failure/LinAlgError: reset predictor and retry once (persist=False)
              * if still failing: skip frame (no rows) to keep run alive and avoid semantic drift
        """

        if bbox_mode and (bbox_csv_out is None):
            raise ValueError("bbox_mode=True requires bbox_csv_out")
        if seg_mode and (seg_csv_out is None):
            raise ValueError("seg_mode=True requires seg_csv_out")

        flush_every_n_frames = int(config_utils._safe_get_config("flush_every_n_frames", flush_every_n_frames))  # type: ignore # noqa:E501
        progress_log_every_sec = float(config_utils._safe_get_config("progress_log_every_sec", 5.0))  # type: ignore

        # Use thread-local models if not provided
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

        # Progress bar
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

        def _opencv_loop_from_current_frame() -> None:
            """
            Continue processing via per-frame OpenCV loop.
            Resumes at the current `frame_count` (i.e., next unread frame).
            """
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
                            # Reset predictor and retry once
                            if _is_linalg_error(e):
                                logger.warning(f"[{job_label}][Frame {frame_count}] SEG LinAlgError; resetting predictor and retrying. err={e}")  # noqa: E501
                            else:
                                logger.warning(f"[{job_label}][Frame {frame_count}] SEG track failed; resetting predictor and retrying. err={e}")  # noqa: E501

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
                                # Skip this frame (don’t switch to predict(), avoids semantic drift)
                                logger.warning(f"[{job_label}][Frame {frame_count}] SEG retry failed; skipping frame. err={e2}")  # noqa: E501
                                seg_results = None

                        if seg_results is not None:
                            r = seg_results[0]
                            boxes = getattr(r, "boxes", None)
                            masks = getattr(r, "masks", None)

                            if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:  # noqa: E501
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
                                        tid = _map_id(raw_tid, seg_id_map, seg_next) if remap_track_ids_per_segment else raw_tid  # noqa: E501
                                        seg_buf.append([int(cls_list[i]), " ".join(flat), int(tid), float(conf_list[i]), frame_count])  # noqa: E501

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
                            if _is_linalg_error(e):
                                logger.warning(f"[{job_label}][Frame {frame_count}] BBOX LinAlgError; resetting predictor and retrying. err={e}")  # noqa: E501
                            else:
                                logger.warning(f"[{job_label}][Frame {frame_count}] BBOX track failed; resetting predictor and retrying. err={e}")  # noqa: E501

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
                                logger.warning(f"[{job_label}][Frame {frame_count}] BBOX retry failed; skipping frame. err={e2}")  # noqa: E501
                                bbox_results = None

                        if bbox_results is not None:
                            r = bbox_results[0]
                            boxes = getattr(r, "boxes", None)

                            if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:  # noqa: E501
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
                                    tid = _map_id(raw_tid, bbox_id_map, bbox_next) if remap_track_ids_per_segment else raw_tid  # noqa: E501
                                    bbox_buf.append([int(c), float(x), float(y), float(w), float(h), int(tid), float(confv), frame_count])  # noqa: E501

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
                # Snellius streaming (fast) — single-mode only; if both bbox+seg, use OpenCV loop
                if self.snellius_mode:
                    if bbox_mode and seg_mode:
                        logger.info(f"[{job_label}] Snellius: bbox+seg enabled -> using OpenCV per-frame loop for robustness.")  # noqa: E501
                        _opencv_loop_from_current_frame()
                        return

                    last_log_t = time.time()

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

                            def _iter_results(it):
                                for rr in it:
                                    if isinstance(rr, (list, tuple)):
                                        yield (rr[0] if rr else None)
                                    else:
                                        yield rr

                            for r in _iter_results(bbox_iter):
                                if r is None:
                                    continue
                                frame_count += 1

                                boxes = getattr(r, "boxes", None)
                                if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:  # noqa: E501
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

                                    for (x, y, w, h), c, raw_tid, confv in zip(xywhn, cls_list, raw_id_list, conf_list):  # noqa: E501
                                        raw_tid = int(raw_tid) if raw_tid is not None else -1
                                        tid = _map_id(raw_tid, bbox_id_map, bbox_next) if remap_track_ids_per_segment else raw_tid  # noqa: E501
                                        bbox_buf.append([int(c), float(x), float(y), float(w), float(h), int(tid), float(confv), frame_count])  # noqa: E501

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

                            def _iter_results(it):
                                for rr in it:
                                    if isinstance(rr, (list, tuple)):
                                        yield (rr[0] if rr else None)
                                    else:
                                        yield rr

                            for r in _iter_results(seg_iter):
                                if r is None:
                                    continue
                                frame_count += 1

                                boxes = getattr(r, "boxes", None)
                                masks = getattr(r, "masks", None)

                                if boxes is not None and getattr(boxes, "xywhn", None) is not None and boxes.xywhn.size(0) > 0:  # noqa: E501
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
                                            tid = _map_id(raw_tid, seg_id_map, seg_next) if remap_track_ids_per_segment else raw_tid  # noqa: E501
                                            seg_buf.append([int(cls_list[i]), " ".join(flat), int(tid), float(conf_list[i]), frame_count])  # noqa: E501

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
                        if _is_linalg_error(e):
                            logger.warning(
                                f"[{job_label}] Snellius streaming crashed with LinAlgError at processed_frames={frame_count}. "  # noqa: E501
                                f"Reset + fallback to OpenCV. err={e}"
                            )
                        else:
                            logger.warning(
                                f"[{job_label}] Snellius streaming crashed at processed_frames={frame_count}. "
                                f"Reset + fallback to OpenCV. err={e}"
                            )

                        # Critical: reset predictor cleanly before continuing
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

                # Non-Snellius: OpenCV loop
                _opencv_loop_from_current_frame()
                return

            finally:
                if pbar is not None:
                    pbar.close()

                # cleanup temp tracker configs
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
