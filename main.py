# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
# -----------------------------------------------------------------------------
# Multithreaded scheduler with bounded video prefetch:
# - Same mapping loop + metadata update behavior.
# - Same video retrieval logic (FTP-first, SSD refresh, YouTube fallback).
# - Segment processing is concurrent via ONE ThreadPoolExecutor per pass.
#   This allows segments from different videos to run concurrently.
# - Video downloads are overlapped with tracking via a separate download executor.
# - Prefetch policy: keep up to max(max_workers+1, prefetch_videos) videos "in flight"
#   (downloading or downloaded+pending processing), so one video is always reserved for
#   the next free worker.
# - Per-video concurrency cap: max_active_segments_per_video (default=1) so "N workers"
#   tends to mean "N different videos at a time".
# - Models are handled inside helper.tracking_mode_threadsafe().
# - CSV outputs are written to TEMP during processing and ATOMICALLY moved to final:
#     data[-1]/__tmp__/bbox|seg/<name>.partial  ->  data[-1]/bbox|seg/<name>.csv
#   only when the whole segment finishes successfully.
# - Adds tqdm trimming progress bar (ffmpeg -progress pipe:1).
# - IMPORTANT: If any CSV exists for (vid, start_time) in data folders, that segment is
#   treated as DONE (regardless of FPS) and WILL NOT be reprocessed nor trigger download.
#
# Snellius/HPC mode additions (behind snellius_mode flag):
# - Shard work by VIDEO across Slurm tasks (avoid redundant staging).
# - Stage videos and trims to node-local $TMPDIR when available.
# - Force one segment worker per process/GPU (scale via Slurm tasks/arrays).
# - Disable sleep/git-pull/email by default in batch jobs.
# -----------------------------------------------------------------------------

import ast
import glob
import math
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from helper_script import Youtube_Helper
from logmod import logs


# -----------------------------------------------------------------------------
# Logging & global tqdm lock (important for multi-threaded progress bars)
# -----------------------------------------------------------------------------
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

helper = Youtube_Helper()

# Make tqdm updates thread-safe across worker threads
tqdm.set_lock(threading.RLock())


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_get_config(key: str, default=None):
    try:
        return common.get_configs(key)
    except Exception:
        return default


def _ensure_dirs(data_path: str):
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.join(data_path, "bbox"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "seg"), exist_ok=True)


def _ensure_tmp_dirs(data_path: str) -> Tuple[str, str]:
    """
    Create temp dirs used for "write-then-commit" outputs.
    Returns (tmp_bbox_dir, tmp_seg_dir).
    """
    tmp_root = os.path.join(data_path, "__tmp__")
    tmp_bbox_dir = os.path.join(tmp_root, "bbox")
    tmp_seg_dir = os.path.join(tmp_root, "seg")
    os.makedirs(tmp_bbox_dir, exist_ok=True)
    os.makedirs(tmp_seg_dir, exist_ok=True)
    return tmp_bbox_dir, tmp_seg_dir


def _cleanup_stale_partials(tmp_dir: str):
    """
    Best-effort cleanup of stale *.partial files from previous crashes.
    """
    try:
        for p in glob.glob(os.path.join(tmp_dir, "*.partial")):
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def _parse_bracket_list(s: str) -> List[str]:
    # mapping.csv uses strings like "[id1,id2]" (sometimes NaN/None)
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    return [x.strip() for x in s.strip().strip("[]").split(",") if x.strip()]


def _index_existing_outputs(data_folders: List[str], want_bbox: bool, want_seg: bool) -> Dict[str, set]:
    """
    Index existing outputs once per pass (FAST; avoids per-segment glob).

    We index by (vid, start_time) ignoring FPS so that if:
        data/*/bbox/{vid}_{start}_*.csv exists
    then we treat that segment as DONE for bbox mode.

    Returns:
      {
        "bbox_start": set((vid, start_int)),
        "seg_start":  set((vid, start_int)),
      }
    """
    bbox_start: set = set()
    seg_start: set = set()

    def _scan_dir(base_folder: str, sub: str, out_set: set):
        d = os.path.join(base_folder, sub)
        if not os.path.isdir(d):
            return
        try:
            with os.scandir(d) as it:
                for e in it:
                    if not e.is_file():
                        continue
                    name = e.name
                    if not name.endswith(".csv"):
                        continue
                    stem = name[:-4]  # drop .csv
                    try:
                        # video IDs can include "_" so split from the right
                        vid_part, st_str, _fps_str = stem.rsplit("_", 2)
                        st_i = int(st_str)
                    except Exception:
                        continue
                    out_set.add((vid_part, st_i))
        except FileNotFoundError:
            return

    for folder in data_folders:
        folder = os.path.abspath(folder)
        if want_bbox:
            _scan_dir(folder, "bbox", bbox_start)
        if want_seg:
            _scan_dir(folder, "seg", seg_start)

    return {"bbox_start": bbox_start, "seg_start": seg_start}


def _fps_is_bad(fps) -> bool:
    return fps is None or fps == 0 or (isinstance(fps, float) and math.isnan(fps))


def _log_run_banner(config: SimpleNamespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    logger.info("============================================================")
    logger.info("Pipeline run configuration")
    logger.info(f"  device={device} gpu={gpu_name}")
    logger.info(f"  cpu_count={os.cpu_count()} torch_num_threads={torch.get_num_threads()}")
    logger.info(f"  snellius_mode={bool(getattr(config, 'snellius_mode', False))}")
    logger.info(
        "  slurm_task_id=%s slurm_task_count=%s tmpdir=%s",
        getattr(config, "slurm_task_id", 0),
        getattr(config, "slurm_task_count", 1),
        getattr(config, "tmpdir", ""),
    )
    logger.info(f"  max_workers={config.max_workers} (segment ThreadPoolExecutor)")
    logger.info(f"  download_workers={config.download_workers} (download ThreadPoolExecutor)")
    logger.info(f"  prefetch_videos={config.prefetch_videos} (max in-flight videos)")
    logger.info(f"  max_active_segments_per_video={config.max_active_segments_per_video}")
    logger.info(f"  active_threads_now={threading.active_count()}")
    logger.info(f"  tracking_mode={config.tracking_mode} segmentation_mode={config.segmentation_mode}")
    logger.info(f"  tracking_model={config.tracking_model}")
    logger.info(f"  segment_model={config.segment_model}")
    logger.info(f"  bbox_tracker={config.bbox_tracker}")
    logger.info(f"  seg_tracker={config.seg_tracker}")
    logger.info(f"  track_buffer_sec={config.track_buffer_sec}")
    logger.info(f"  confidence={getattr(helper, 'confidence', 0.0)}")
    logger.info(f"  external_ssd={config.external_ssd} compress_youtube_video={config.compress_youtube_video}")
    logger.info(f"  delete_youtube_video={config.delete_youtube_video}")
    logger.info("============================================================")


def _copy_to_ssd_if_needed(base_video_path: str, internal_ssd: str, vid: str) -> str:
    """
    Preserves the prior 'copy_video_safe' behavior.
    Ensures the file is on SSD and returns the SSD path.
    """
    if os.path.dirname(base_video_path) == internal_ssd:
        return base_video_path

    out = helper.copy_video_safe(base_video_path, internal_ssd, vid)
    logger.debug(f"Copied to {out}.")
    return os.path.join(internal_ssd, f"{vid}.mp4")


def _hms_to_seconds(hms: str) -> Optional[float]:
    # ffmpeg out_time example: "00:00:12.345678"
    try:
        parts = hms.strip().split(":")
        if len(parts) != 3:
            return None
        hh = float(parts[0])
        mm = float(parts[1])
        ss = float(parts[2])
        return hh * 3600.0 + mm * 60.0 + ss
    except Exception:
        return None


def _trim_video_with_progress(
    input_path: str,
    output_path: str,
    start_time: int,
    end_time: int,
    job_label: str,
    tqdm_position: int,
):
    """
    Trim with a tqdm progress bar using ffmpeg -progress pipe:1.

    Strategy:
      1) Try stream-copy (fast). If progress doesn't move (common), retry with re-encode.
      2) Re-encode guarantees progress updates but is slower.
    """
    duration = max(0.0, float(end_time) - float(start_time))
    if duration <= 0.0:
        helper.trim_video(input_path, output_path, start_time, end_time)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _run_ffmpeg(cmd: List[str]) -> float:
        pbar = tqdm(
            total=duration,
            desc=f"trim: {job_label}",
            unit="s",
            dynamic_ncols=True,
            position=tqdm_position,
            leave=False,  # shows while trimming; then yields same line to frames pbar
        )
        last_sec = 0.0
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("out_time_ms="):
                        try:
                            ms = int(line.split("=", 1)[1].strip())
                            cur = ms / 1_000_000.0
                        except Exception:
                            continue
                        cur = min(cur, duration)
                        if cur > last_sec:
                            pbar.update(cur - last_sec)
                            last_sec = cur

                    elif line.startswith("out_time="):
                        cur = _hms_to_seconds(line.split("=", 1)[1].strip())
                        if cur is None:
                            continue
                        cur = min(cur, duration)
                        if cur > last_sec:
                            pbar.update(cur - last_sec)
                            last_sec = cur

                    elif line.startswith("progress=") and line.endswith("end"):
                        break

            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)

            if last_sec < duration:
                pbar.update(duration - last_sec)

            return last_sec

        finally:
            try:
                pbar.close()
            except Exception:
                pass

    # 1) Fast path: stream copy
    cmd_copy = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_time),
        "-to",
        str(end_time),
        "-i",
        input_path,
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        output_path,
    ]

    try:
        progressed = _run_ffmpeg(cmd_copy)

        # If no visible progress, retry with re-encode for a real bar
        if progressed < 1.0:
            logger.info(f"[trim] no progress from stream-copy; retrying with re-encode for: {job_label}")

            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass

            cmd_reencode = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                "-progress",
                "pipe:1",
                "-nostats",
                output_path,
            ]
            _run_ffmpeg(cmd_reencode)

    except Exception as e:
        logger.warning(f"[trim] ffmpeg trim failed; falling back to helper.trim_video(). reason={e!r}")
        helper.trim_video(input_path, output_path, start_time, end_time)


def _ensure_video_available(
    vid: str,
    config: SimpleNamespace,
    secret: SimpleNamespace,
    output_path: str,
    video_paths: List[str],
) -> Tuple[str, str, str, int, bool]:
    """
    Preserves your prior retrieval logic.
    Returns: (base_video_path, video_title, resolution, video_fps, ftp_download)
    """
    ftp_download = False
    resolution = "unknown"
    video_title = vid

    base_video_path = os.path.join(output_path, f"{vid}.mp4")

    if config.external_ssd:
        existing_path = os.path.join(output_path, f"{vid}.mp4")

        if os.path.exists(existing_path):
            tmp_dir = os.path.join(output_path, "__tmp_dl")
            os.makedirs(tmp_dir, exist_ok=True)

            logger.info(f"{vid}: SSD copy exists; attempting FTP refresh into temp={tmp_dir}.")
            tmp_result = helper.download_videos_from_ftp(
                filename=vid,
                base_url=config.ftp_server,
                out_dir=tmp_dir,
                username=secret.ftp_username,
                password=secret.ftp_password,
            )

            if tmp_result:
                tmp_video_path, video_title, resolution, video_fps = tmp_result
                if _fps_is_bad(video_fps):
                    logger.warning(f"{vid}: invalid FPS in refreshed file; keeping existing SSD copy.")
                    try:
                        os.remove(tmp_video_path)
                    except Exception:
                        pass
                    video_fps2 = helper.get_video_fps(existing_path)
                    if _fps_is_bad(video_fps2):
                        raise RuntimeError(f"{vid}: existing SSD copy also has invalid FPS.")
                    helper.set_video_title(video_title)
                    return existing_path, video_title, resolution, int(video_fps2), False  # type: ignore

                try:
                    os.remove(existing_path)
                except FileNotFoundError:
                    pass

                final_path = os.path.join(output_path, f"{vid}.mp4")
                shutil.move(tmp_video_path, final_path)
                ftp_download = True
                helper.set_video_title(video_title)

                logger.info(f"{vid}: refreshed from FTP and replaced SSD copy at {final_path}.")
                return final_path, video_title, resolution, int(video_fps), ftp_download

            logger.info(f"{vid}: FTP not available; using existing SSD copy.")
            helper.set_video_title(video_title)
            video_fps = helper.get_video_fps(existing_path)

            if _fps_is_bad(video_fps):
                logger.warning(f"{vid}: invalid FPS on SSD copy; attempting YouTube fallback.")
                yt_result = helper.download_video_with_resolution(vid=vid, output_path=output_path)
                if not yt_result:
                    raise RuntimeError(f"{vid}: YouTube fallback failed and SSD copy FPS invalid.")
                video_file_path, video_title, resolution, video_fps = yt_result
                if _fps_is_bad(video_fps):
                    raise RuntimeError(f"{vid}: YouTube fallback produced invalid FPS.")
                helper.set_video_title(video_title)
                return video_file_path, video_title, resolution, int(video_fps), False

            return existing_path, video_title, resolution, int(video_fps), False  # type: ignore

        logger.info(f"{vid}: no SSD copy; attempting FTP download to {output_path}.")
        result = helper.download_videos_from_ftp(
            filename=vid,
            base_url=config.ftp_server,
            out_dir=output_path,
            username=secret.ftp_username,
            password=secret.ftp_password,
        )
        if result:
            ftp_download = True
        if result is None:
            logger.info(f"{vid}: FTP not found/failed; attempting YouTube download.")
            result = helper.download_video_with_resolution(vid=vid, output_path=output_path)

        if not result:
            raise RuntimeError(f"{vid}: forced download failed (FTP+fallback).")

        video_file_path, video_title, resolution, video_fps = result
        if _fps_is_bad(video_fps):
            raise RuntimeError(f"{vid}: invalid video_fps after download.")
        helper.set_video_title(video_title)

        logger.info(f"{vid}: downloaded successfully. res={resolution} fps={int(video_fps)} path={video_file_path}")
        return video_file_path, video_title, resolution, int(video_fps), ftp_download

    exists_somewhere = any(os.path.exists(os.path.join(path, f"{vid}.mp4")) for path in video_paths)

    if not exists_somewhere:
        logger.info(f"{vid}: not cached; attempting FTP download to {output_path}.")
        result = helper.download_videos_from_ftp(
            filename=vid,
            base_url=config.ftp_server,
            out_dir=output_path,
            username=secret.ftp_username,
            password=secret.ftp_password,
        )
        if result:
            ftp_download = True
        if result is None:
            logger.info(f"{vid}: FTP not found/failed; attempting YouTube download.")
            result = helper.download_video_with_resolution(vid=vid, output_path=output_path)

        if result:
            video_file_path, video_title, resolution, video_fps = result
            if _fps_is_bad(video_fps):
                raise RuntimeError(f"{vid}: invalid video_fps after download.")
            helper.set_video_title(video_title)
            logger.info(
                f"{vid}: downloaded successfully. res={resolution} fps={int(video_fps)} path={video_file_path}"
            )
            return video_file_path, video_title, resolution, int(video_fps), ftp_download

        if os.path.exists(base_video_path):
            helper.set_video_title(video_title)
            video_fps = helper.get_video_fps(base_video_path)
            if _fps_is_bad(video_fps):
                raise RuntimeError(f"{vid}: invalid FPS on local fallback file.")
            logger.info(f"{vid}: found locally at {base_video_path}. fps={int(video_fps)}")  # type: ignore
            return base_video_path, video_title, resolution, int(video_fps), False  # type: ignore

        raise RuntimeError(f"{vid}: video not found and download failed.")

    existing_folder = next((p for p in video_paths if os.path.exists(os.path.join(p, f"{vid}.mp4"))), None)
    use_folder = existing_folder if existing_folder else video_paths[-1]
    base_video_path = os.path.join(use_folder, f"{vid}.mp4")
    helper.set_video_title(video_title)

    video_fps = helper.get_video_fps(base_video_path)
    if _fps_is_bad(video_fps):
        raise RuntimeError(f"{vid}: invalid FPS on cached file.")

    logger.info(f"{vid}: using cached video at {base_video_path}. fps={int(video_fps)}")  # type: ignore
    return base_video_path, video_title, resolution, int(video_fps), False  # type: ignore


@dataclass
class VideoReq:
    vid: str
    segments: List[Tuple[int, int]]
    city: str = ""
    state: str = ""
    country: str = ""
    iso3: str = ""


@dataclass
class VideoCtx:
    vid: str
    base_video_path: str
    fps: int
    resolution: str
    ftp_download: bool
    output_path: str
    external_ssd: bool
    delete_youtube_video: bool
    pending: int = 0
    processed_any: bool = False


@dataclass
class DownloadResult:
    vid: str
    base_video_path: str
    fps: int
    resolution: str
    ftp_download: bool
    # job tuple:
    # (vid, st, et, base_video_path, run_bbox, run_seg,
    #  bbox_final, seg_final, bbox_tmp, seg_tmp, fps)
    segment_jobs: List[
        Tuple[
            str,
            int,
            int,
            str,
            bool,
            bool,
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
            int,
        ]
    ]
    elapsed_sec: float


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    counter_processed = 0  # ensure defined for crash email

    try:
        # ---------------------------------------------------------------------
        # Load configuration
        # ---------------------------------------------------------------------
        max_workers = max(1, int(_safe_get_config("max_workers", 3)))  # type: ignore[arg-type]
        download_workers = max(1, int(_safe_get_config("download_workers", 1)))  # type: ignore[arg-type]
        max_active_segments_per_video = max(
            1, int(_safe_get_config("max_active_segments_per_video", 1))
        )  # type: ignore[arg-type]

        prefetch_raw = int(_safe_get_config("prefetch_videos", 0) or 0)
        prefetch_videos = prefetch_raw if prefetch_raw > 0 else (max_workers + 1)
        prefetch_videos = max(prefetch_videos, max_workers + 1)  # enforce N+1

        config = SimpleNamespace(
            mapping=common.get_configs("mapping"),
            videos=common.get_configs("videos"),
            data=common.get_configs("data"),
            countries_analyse=_safe_get_config("countries_analyse", []),
            update_pop_country=_safe_get_config("update_pop_country", False),
            update_gini_value=_safe_get_config("update_gini_value", False),
            segmentation_mode=_safe_get_config("segmentation_mode", False),
            tracking_mode=_safe_get_config("tracking_mode", False),
            delete_youtube_video=_safe_get_config("delete_youtube_video", False),
            compress_youtube_video=_safe_get_config("compress_youtube_video", False),
            external_ssd=_safe_get_config("external_ssd", False),
            ftp_server=_safe_get_config("ftp_server", None),
            tracking_model=_safe_get_config("tracking_model", ""),
            segment_model=_safe_get_config("segment_model", ""),
            bbox_tracker=_safe_get_config("bbox_tracker", ""),
            seg_tracker=_safe_get_config("seg_tracker", ""),
            track_buffer_sec=_safe_get_config("track_buffer_sec", 1),
            save_annotated_video=_safe_get_config("save_annotated_video", False),
            sleep_sec=_safe_get_config("sleep_sec", 0),
            git_pull=_safe_get_config("git_pull", False),
            machine_name=_safe_get_config("machine_name", "unknown"),
            email_send=_safe_get_config("email_send", False),
            email_sender=_safe_get_config("email_sender", ""),
            email_recipients=_safe_get_config("email_recipients", []),
            max_workers=int(max_workers),
            download_workers=int(download_workers),
            prefetch_videos=int(prefetch_videos),
            max_active_segments_per_video=int(max_active_segments_per_video),
            # Snellius/HPC master flag (default False so local behavior is unchanged)
            snellius_mode=bool(_safe_get_config("snellius_mode", False)),
        )

        # ---------------------------------------------------------------------
        # Snellius/Slurm context + runtime overrides (only when snellius_mode=True)
        # ---------------------------------------------------------------------
        config.slurm_task_id = int(os.getenv("SLURM_PROCID", os.getenv("SLURM_ARRAY_TASK_ID", "0")))
        config.slurm_task_count = int(os.getenv("SLURM_NTASKS", os.getenv("SLURM_ARRAY_TASK_COUNT", "1")))
        config.tmpdir = os.getenv("TMPDIR", "") or ""

        if bool(config.snellius_mode):
            # Scale via Slurm tasks/arrays (one process per GPU), not via in-process concurrency.
            config.max_workers = 1
            config.download_workers = 1
            config.max_active_segments_per_video = 1
            config.prefetch_videos = max(2, int(config.prefetch_videos or 2))

            # Avoid idle looping/auto-updating in batch jobs
            config.sleep_sec = 0
            config.git_pull = False

            # Generally undesirable in batch jobs (can be re-enabled if you want)
            config.email_send = False

            # Avoid accidental deletion of shared /projects videos
            config.delete_youtube_video = False

            # TMPDIR is our fast work area; do not engage "SSD refresh" semantics.
            config.external_ssd = False

        # ---------------------------------------------------------------------
        # Load secrets
        # ---------------------------------------------------------------------
        secret = SimpleNamespace(
            email_smtp=common.get_secrets("email_smtp"),
            email_account=common.get_secrets("email_account"),
            email_password=common.get_secrets("email_password"),
            ftp_username=common.get_secrets("ftp_username"),
            ftp_password=common.get_secrets("ftp_password"),
        )

        _log_run_banner(config)

        pass_index = 0

        while True:
            pass_index += 1
            pass_start_ts = time.time()

            mapping = pd.read_csv(config.mapping)
            video_paths = config.videos

            if config.external_ssd:
                internal_ssd = config.videos[-1]
                os.makedirs(internal_ssd, exist_ok=True)
                output_path = config.videos[-1]
            else:
                internal_ssd = None
                output_path = config.videos[-1]

            # Snellius: stage videos + trims to node-local scratch when available
            scratch_videos_dir: Optional[str] = None
            if bool(config.snellius_mode) and getattr(config, "tmpdir", ""):
                scratch_videos_dir = os.path.join(str(config.tmpdir), "videos")
                os.makedirs(scratch_videos_dir, exist_ok=True)

            data_folders = config.data
            data_path = config.data[-1]
            countries_analyse = config.countries_analyse or []
            counter_processed = 0

            logger.info(
                f"=== Pass {pass_index} started === rows={mapping.shape[0]} "
                f"max_workers={config.max_workers} prefetch_videos={config.prefetch_videos} "
                f"download_workers={config.download_workers} active_threads_now={threading.active_count()} "
                f"tracking_mode={bool(config.tracking_mode)} seg_mode={bool(config.segmentation_mode)}"
            )

            if config.update_pop_country:
                logger.info("Updating population in mapping file...")
                helper.update_population_in_csv(mapping)

            if config.update_gini_value:
                logger.info("Updating GINI values in mapping file...")
                helper.fill_gini_data(mapping)

            helper.delete_folder(folder_path="runs")
            _ensure_dirs(data_path)

            # temp output dirs (write-then-commit)
            tmp_bbox_dir, tmp_seg_dir = _ensure_tmp_dirs(data_path)
            _cleanup_stale_partials(tmp_bbox_dir)
            _cleanup_stale_partials(tmp_seg_dir)

            seg_mode_cfg = bool(config.segmentation_mode)
            bbox_mode_cfg = bool(config.tracking_mode)

            if not (seg_mode_cfg or bbox_mode_cfg):
                logger.info("Both tracking_mode and segmentation_mode are disabled; nothing to do.")
                break

            # Index existing outputs ONCE per pass (skip work + skip downloads)
            existing_idx = _index_existing_outputs(
                data_folders=data_folders,
                want_bbox=bbox_mode_cfg,
                want_seg=seg_mode_cfg,
            )
            bbox_done_start = existing_idx["bbox_start"]
            seg_done_start = existing_idx["seg_start"]
            logger.info(
                f"Existing outputs indexed: bbox_start={len(bbox_done_start)} seg_start={len(seg_done_start)} "
                f"(done by (vid,start) ignoring fps)"
            )

            # -----------------------------------------------------------------
            # Stage 1: Build per-video requests
            # -----------------------------------------------------------------
            req_by_vid: Dict[str, VideoReq] = {}

            pbar_rows = tqdm(
                mapping.iterrows(),
                total=mapping.shape[0],
                desc="Mapping rows",
                dynamic_ncols=True,
                position=0,
                leave=True,
            )

            for _, row in pbar_rows:
                video_ids = _parse_bracket_list(str(row.get("videos", "")))
                if not video_ids:
                    continue

                iso3 = str(row.get("iso3", ""))
                if countries_analyse and iso3 and iso3 not in countries_analyse:
                    continue

                city = str(row.get("city", ""))
                state = str(row.get("state", ""))
                country = str(row.get("country", ""))

                try:
                    start_times = ast.literal_eval(row["start_time"])
                    end_times = ast.literal_eval(row["end_time"])
                except Exception:
                    logger.warning("Failed to parse start_time/end_time for a row; skipping row.")
                    continue

                if not isinstance(start_times, list) or not isinstance(end_times, list):
                    continue

                for i, vid in enumerate(video_ids):
                    st_list = start_times[i] if i < len(start_times) else []
                    et_list = end_times[i] if i < len(end_times) else []
                    if not isinstance(st_list, list) or not isinstance(et_list, list):
                        continue

                    needed_segments: List[Tuple[int, int]] = []
                    for st, et in zip(st_list, et_list):
                        st_i = int(st)
                        et_i = int(et)

                        # DONE logic ignores FPS (per your requirement)
                        bbox_done = (not bbox_mode_cfg) or ((vid, st_i) in bbox_done_start)
                        seg_done = (not seg_mode_cfg) or ((vid, st_i) in seg_done_start)

                        if bbox_done and seg_done:
                            continue

                        needed_segments.append((st_i, et_i))

                    if not needed_segments:
                        continue

                    if vid not in req_by_vid:
                        req_by_vid[vid] = VideoReq(
                            vid=vid,
                            segments=[],
                            city=city,
                            state=state,
                            country=country,
                            iso3=iso3,
                        )

                    existing = set(req_by_vid[vid].segments)
                    for seg in needed_segments:
                        if seg not in existing:
                            req_by_vid[vid].segments.append(seg)
                            existing.add(seg)

            pbar_rows.close()

            video_reqs = list(req_by_vid.values())
            logger.info(f"Pass {pass_index}: videos needing work={len(video_reqs)} (after done-index pre-check).")

            # Snellius: shard by VIDEO across Slurm tasks to avoid redundant staging/I/O
            if bool(config.snellius_mode) and int(getattr(config, "slurm_task_count", 1)) > 1:
                task_id = int(getattr(config, "slurm_task_id", 0))
                task_count = int(getattr(config, "slurm_task_count", 1))
                video_reqs = sorted(video_reqs, key=lambda r: r.vid)
                video_reqs = [vr for i, vr in enumerate(video_reqs) if (i % task_count) == task_id]
                logger.info(
                    "Snellius sharding applied: task_id=%d task_count=%d videos_assigned=%d",
                    task_id,
                    task_count,
                    len(video_reqs),
                )

            # -----------------------------------------------------------------
            # Stage 2: Bounded prefetch + global segment pool with per-video cap
            # -----------------------------------------------------------------
            ctx_lock = threading.Lock()
            ctx_by_vid: Dict[str, VideoCtx] = {}

            ready_jobs_by_vid: Dict[
                str,
                List[
                    Tuple[
                        str,
                        int,
                        int,
                        str,
                        bool,
                        bool,
                        Optional[str],
                        Optional[str],
                        Optional[str],
                        Optional[str],
                        int,
                    ]
                ],
            ] = {}
            active_segments_by_vid: Dict[str, int] = {}
            rr_vids: List[str] = []
            rr_state = {"idx": 0}

            pbar_segs = tqdm(
                total=0,
                desc="Segments completed",
                unit="seg",
                dynamic_ncols=True,
                position=1,
                leave=True,
            )

            def _maybe_add_rr_vid(vid: str):
                if vid in rr_vids:
                    return
                if ready_jobs_by_vid.get(vid):
                    rr_vids.append(vid)

            def _maybe_remove_rr_vid(vid: str):
                if vid not in rr_vids:
                    return
                try:
                    idx = rr_vids.index(vid)
                except ValueError:
                    return
                rr_vids.pop(idx)
                if rr_vids:
                    rr_state["idx"] %= len(rr_vids)
                else:
                    rr_state["idx"] = 0

            def _get_or_create_ctx(vid: str, base_video_path: str, fps: int, resolution: str, ftp_download: bool) -> VideoCtx:
                with ctx_lock:
                    if vid not in ctx_by_vid:
                        ctx_by_vid[vid] = VideoCtx(
                            vid=vid,
                            base_video_path=base_video_path,
                            fps=int(fps),
                            resolution=resolution,
                            ftp_download=bool(ftp_download),
                            output_path=output_path,
                            # On Snellius, base_video_path may be staged into TMPDIR;
                            # treat it like an SSD work copy so it gets cleaned.
                            external_ssd=bool(
                                config.external_ssd or (bool(config.snellius_mode) and bool(scratch_videos_dir))
                            ),
                            delete_youtube_video=bool(config.delete_youtube_video),
                            pending=0,
                            processed_any=False,
                        )
                        active_segments_by_vid[vid] = 0
                    else:
                        ctx_by_vid[vid].base_video_path = base_video_path
                        ctx_by_vid[vid].fps = int(fps)
                        ctx_by_vid[vid].resolution = resolution
                        ctx_by_vid[vid].ftp_download = bool(ftp_download)
                    return ctx_by_vid[vid]

            def _inflight_video_count(inflight_downloads: Dict[Any, VideoReq]) -> int:
                with ctx_lock:
                    return len(inflight_downloads) + len(ctx_by_vid)

            def _finalize_video_if_done(vid: str):
                with ctx_lock:
                    ctx = ctx_by_vid.get(vid)
                    if not ctx:
                        return

                    pending_now = ctx.pending
                    processed_any = ctx.processed_any
                    active_now = active_segments_by_vid.get(vid, 0)
                    has_ready = bool(ready_jobs_by_vid.get(vid))

                    if pending_now > 0 or active_now > 0 or has_ready:
                        return

                    ctx_snapshot = VideoCtx(**ctx.__dict__)

                    ctx_by_vid.pop(vid, None)
                    active_segments_by_vid.pop(vid, None)
                    ready_jobs_by_vid.pop(vid, None)
                    _maybe_remove_rr_vid(vid)

                if not processed_any:
                    return

                if ctx_snapshot.external_ssd:
                    try:
                        os.remove(ctx_snapshot.base_video_path)
                        logger.info(f"{vid}: cleaned SSD/scratch working copy: {ctx_snapshot.base_video_path}")
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        logger.warning(f"{vid}: failed SSD/scratch cleanup: {e}")

                if ctx_snapshot.ftp_download:
                    try:
                        os.remove(ctx_snapshot.base_video_path)
                        logger.info(f"{vid}: removed FTP-downloaded working copy: {ctx_snapshot.base_video_path}")
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        logger.warning(f"{vid}: failed FTP cleanup: {e}")

                if ctx_snapshot.delete_youtube_video:
                    try:
                        os.remove(os.path.join(ctx_snapshot.output_path, f"{vid}.mp4"))
                        logger.info(f"{vid}: deleted YouTube video due to delete_youtube_video=True")
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        logger.warning(f"{vid}: failed delete_youtube_video cleanup: {e}")

            def _download_and_prepare(req: VideoReq) -> DownloadResult:
                """
                Download/locate the video, compute fps-specific output filenames,
                and return only the segment jobs that are still needed.

                NOTE: Uses bbox_done_start/seg_done_start (done by (vid,start) ignoring fps),
                so we do NOT download/redo segments already completed.
                """
                t0 = time.time()
                vid = req.vid

                base_video_path, _title, resolution, video_fps, ftp_download = _ensure_video_available(
                    vid=vid,
                    config=config,
                    secret=secret,
                    output_path=output_path,
                    video_paths=video_paths,
                )

                # Snellius: stage to node-local scratch (preferred)
                if bool(config.snellius_mode) and scratch_videos_dir:
                    base_video_path = _copy_to_ssd_if_needed(base_video_path, scratch_videos_dir, vid)
                elif config.external_ssd and internal_ssd:
                    base_video_path = _copy_to_ssd_if_needed(base_video_path, internal_ssd, vid)

                fps_i = int(video_fps)

                segment_jobs: List[
                    Tuple[
                        str,
                        int,
                        int,
                        str,
                        bool,
                        bool,
                        Optional[str],
                        Optional[str],
                        Optional[str],
                        Optional[str],
                        int,
                    ]
                ] = []

                for (st, et) in req.segments:
                    st_i = int(st)
                    et_i = int(et)

                    run_bbox = bool(bbox_mode_cfg and ((vid, st_i) not in bbox_done_start))
                    run_seg = bool(seg_mode_cfg and ((vid, st_i) not in seg_done_start))

                    if not run_bbox and not run_seg:
                        continue

                    segment_csv = f"{vid}_{st_i}_{fps_i}.csv"

                    # Final paths (only commit at end)
                    bbox_final = os.path.join(data_path, "bbox", segment_csv) if run_bbox else None
                    seg_final = os.path.join(data_path, "seg", segment_csv) if run_seg else None

                    # Temp paths (written during processing)
                    bbox_tmp = os.path.join(tmp_bbox_dir, segment_csv + ".partial") if run_bbox else None
                    seg_tmp = os.path.join(tmp_seg_dir, segment_csv + ".partial") if run_seg else None

                    segment_jobs.append(
                        (vid, st_i, et_i, base_video_path, run_bbox, run_seg, bbox_final, seg_final, bbox_tmp, seg_tmp, fps_i)
                    )

                dt = time.time() - t0
                return DownloadResult(
                    vid=vid,
                    base_video_path=base_video_path,
                    fps=fps_i,
                    resolution=resolution,
                    ftp_download=bool(ftp_download),
                    segment_jobs=segment_jobs,
                    elapsed_sec=dt,
                )

            def _segment_worker(
                job: Tuple[str, int, int, str, bool, bool, Optional[str], Optional[str], Optional[str], Optional[str], int]
            ) -> str:
                vid, st, et, base_path, do_bbox, do_seg, bbox_final, seg_final, bbox_tmp, seg_tmp, fps = job
                th = threading.current_thread().name
                t0 = time.time()

                mode_str = f"{'bbox' if do_bbox else ''}{'&' if do_bbox and do_seg else ''}{'seg' if do_seg else ''}"
                job_label = f"{vid} [{st}-{et}s] ({mode_str})"

                try:
                    worker_idx = int(th.split("_")[-1])
                except Exception:
                    worker_idx = 0
                tqdm_pos = 2 + (worker_idx % max(1, int(config.max_workers)))

                # Snellius: do trims in TMPDIR as well (fast local disk)
                if bool(config.snellius_mode) and scratch_videos_dir:
                    work_dir = scratch_videos_dir
                else:
                    work_dir = internal_ssd if (config.external_ssd and internal_ssd) else output_path

                trimmed_path = os.path.join(work_dir, f"{vid}_{st}_{et}_mod.mp4")

                logger.info(
                    f"[worker-start] thread={th} job={job_label} fps={int(fps)} "
                    f"trimmed={os.path.basename(trimmed_path)} "
                    f"bbox_final={os.path.basename(bbox_final) if bbox_final else None} "
                    f"seg_final={os.path.basename(seg_final) if seg_final else None}"
                )

                if bbox_tmp:
                    os.makedirs(os.path.dirname(bbox_tmp), exist_ok=True)
                if seg_tmp:
                    os.makedirs(os.path.dirname(seg_tmp), exist_ok=True)

                # Best-effort cleanup of temp partials from previous attempts of same segment
                for p in (bbox_tmp, seg_tmp):
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

                try:
                    # Trim with progress bar
                    logger.info(f"[trim-start] thread={th} job={job_label}")
                    end_time_adj = max(st, et - 1)

                    _trim_video_with_progress(
                        input_path=base_path,
                        output_path=trimmed_path,
                        start_time=st,
                        end_time=end_time_adj,
                        job_label=job_label,
                        tqdm_position=tqdm_pos,
                    )
                    logger.info(f"[trim-done] thread={th} job={job_label}")

                    # Track (per-frame tqdm is inside helper)
                    logger.info(f"[track-start] thread={th} job={job_label}")
                    helper.tracking_mode_threadsafe(
                        input_video_path=trimmed_path,
                        video_fps=int(fps),
                        bbox_mode=do_bbox,
                        seg_mode=do_seg,
                        bbox_csv_out=bbox_tmp,  # write temp
                        seg_csv_out=seg_tmp,    # write temp
                        job_label=job_label,
                        tqdm_position=tqdm_pos,  # reuse worker line
                        show_frame_pbar=(False if bool(config.snellius_mode) else True),
                        postfix_every_n=30,
                    )
                    logger.info(f"[track-done] thread={th} job={job_label}")

                    # Commit temp -> final only AFTER tracking succeeds
                    if do_bbox and bbox_tmp and bbox_final:
                        if os.path.exists(bbox_tmp):
                            os.makedirs(os.path.dirname(bbox_final), exist_ok=True)
                            os.replace(bbox_tmp, bbox_final)  # atomic on same filesystem
                            logger.info(f"[commit] job={job_label} bbox -> {bbox_final}")
                            bbox_done_start.add((vid, st))  # optional: keep in-memory done set updated

                    if do_seg and seg_tmp and seg_final:
                        if os.path.exists(seg_tmp):
                            os.makedirs(os.path.dirname(seg_final), exist_ok=True)
                            os.replace(seg_tmp, seg_final)
                            logger.info(f"[commit] job={job_label} seg -> {seg_final}")
                            seg_done_start.add((vid, st))  # optional: keep in-memory done set updated

                    # Optional log sizes
                    if bbox_final and os.path.exists(bbox_final):
                        logger.info(f"[csv] job={job_label} bbox_bytes={os.path.getsize(bbox_final)}")
                    if seg_final and os.path.exists(seg_final):
                        logger.info(f"[csv] job={job_label} seg_bytes={os.path.getsize(seg_final)}")

                    return vid

                except Exception:
                    # Do NOT commit partials; remove temp outputs on failure
                    for p in (bbox_tmp, seg_tmp):
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                    raise

                finally:
                    try:
                        os.remove(trimmed_path)
                    except FileNotFoundError:
                        pass
                    dt = time.time() - t0
                    logger.info(f"[worker-done] thread={th} job={job_label} elapsed_sec={dt:.2f}")

            def _dispatch_segments(inflight_process: Dict[Any, str], process_pool: ThreadPoolExecutor):
                free_slots = int(config.max_workers) - len(inflight_process)
                if free_slots <= 0:
                    return

                scheduled = 0
                for _ in range(free_slots):
                    if not rr_vids:
                        break

                    tried = 0
                    picked_vid = None

                    while tried < len(rr_vids):
                        idx = rr_state["idx"] % len(rr_vids)
                        vid = rr_vids[idx]
                        rr_state["idx"] = (idx + 1) % len(rr_vids)
                        tried += 1

                        if not ready_jobs_by_vid.get(vid):
                            continue

                        active_now = active_segments_by_vid.get(vid, 0)
                        if active_now >= int(config.max_active_segments_per_video):
                            continue

                        picked_vid = vid
                        break

                    if picked_vid is None:
                        break

                    job = ready_jobs_by_vid[picked_vid].pop(0)
                    if not ready_jobs_by_vid[picked_vid]:
                        _maybe_remove_rr_vid(picked_vid)

                    active_segments_by_vid[picked_vid] = active_segments_by_vid.get(picked_vid, 0) + 1

                    pf = process_pool.submit(_segment_worker, job)
                    inflight_process[pf] = picked_vid
                    scheduled += 1

                if scheduled:
                    logger.info(
                        f"Dispatched {scheduled} segment(s). "
                        f"active_segments={len(inflight_process)}/{config.max_workers} rr_videos={len(rr_vids)}"
                    )

            download_pool = ThreadPoolExecutor(max_workers=int(config.download_workers), thread_name_prefix="DL")
            process_pool = ThreadPoolExecutor(max_workers=int(config.max_workers), thread_name_prefix="SEG")

            inflight_downloads: Dict[Any, VideoReq] = {}
            inflight_process: Dict[Any, str] = {}

            flags = {"done_submitting": False}
            state_lock = threading.Lock()

            try:
                req_iter = iter(video_reqs)

                def _submit_downloads_up_to_prefetch():
                    target_inflight = int(config.prefetch_videos)
                    with state_lock:
                        while (
                            (not flags["done_submitting"])
                            and (len(inflight_downloads) < int(config.download_workers))
                            and (_inflight_video_count(inflight_downloads) < target_inflight)
                        ):
                            try:
                                req = next(req_iter)
                            except StopIteration:
                                flags["done_submitting"] = True
                                break

                            logger.info(
                                f"{req.vid}: queued for download+prepare "
                                f"(segments_requested={len(req.segments)}) "
                                f"inflight_videos={_inflight_video_count(inflight_downloads)+1}/{target_inflight} "
                                f"downloads_inflight={len(inflight_downloads)+1}/{config.download_workers}"
                            )
                            fut = download_pool.submit(_download_and_prepare, req)
                            inflight_downloads[fut] = req

                _submit_downloads_up_to_prefetch()
                first_exception: Optional[BaseException] = None

                while inflight_downloads or inflight_process or (not flags["done_submitting"]):
                    _submit_downloads_up_to_prefetch()
                    _dispatch_segments(inflight_process, process_pool)

                    wait_set = set(inflight_downloads.keys()) | set(inflight_process.keys())
                    if not wait_set:
                        time.sleep(0.05)
                        continue

                    done, _ = wait(wait_set, return_when=FIRST_COMPLETED)

                    for fut in done:
                        # Download finished
                        if fut in inflight_downloads:
                            req = inflight_downloads.pop(fut)
                            try:
                                dr = fut.result()

                                logger.info(
                                    f"{dr.vid}: download+prepare done "
                                    f"fps={dr.fps} res={dr.resolution} ftp_download={dr.ftp_download} "
                                    f"segment_jobs={len(dr.segment_jobs)} elapsed_sec={dr.elapsed_sec:.2f}"
                                )

                                if not dr.segment_jobs:
                                    logger.info(f"{dr.vid}: after prepare, nothing to run; skipping scheduling.")
                                    continue

                                _get_or_create_ctx(dr.vid, dr.base_video_path, dr.fps, dr.resolution, dr.ftp_download)

                                with ctx_lock:
                                    ctx_by_vid[dr.vid].pending += len(dr.segment_jobs)

                                pbar_segs.total += len(dr.segment_jobs)
                                pbar_segs.refresh()

                                ready_jobs_by_vid.setdefault(dr.vid, []).extend(dr.segment_jobs)
                                _maybe_add_rr_vid(dr.vid)

                                _dispatch_segments(inflight_process, process_pool)

                            except BaseException as e:
                                if first_exception is None:
                                    first_exception = e
                                logger.error(f"Download/prepare failed for {req.vid}: {e!r}")
                                break

                        # Segment finished
                        elif fut in inflight_process:
                            vid_done = inflight_process.pop(fut)
                            try:
                                _ = fut.result()
                                counter_processed += 1
                                pbar_segs.update(1)

                                with ctx_lock:
                                    if vid_done in ctx_by_vid:
                                        ctx_by_vid[vid_done].processed_any = True
                                        ctx_by_vid[vid_done].pending = max(0, ctx_by_vid[vid_done].pending - 1)

                                active_segments_by_vid[vid_done] = max(0, active_segments_by_vid.get(vid_done, 0) - 1)

                                _finalize_video_if_done(vid_done)

                                _submit_downloads_up_to_prefetch()
                                _dispatch_segments(inflight_process, process_pool)

                            except BaseException as e:
                                if first_exception is None:
                                    first_exception = e
                                logger.error(f"Worker failed for {vid_done}: {e!r}")
                                break

                    if first_exception is not None:
                        break

                if first_exception is not None:
                    for f in list(inflight_downloads.keys()):
                        try:
                            f.cancel()
                        except Exception:
                            pass
                    for f in list(inflight_process.keys()):
                        try:
                            f.cancel()
                        except Exception:
                            pass
                    raise first_exception

            finally:
                try:
                    pbar_segs.close()
                except Exception:
                    pass

                try:
                    download_pool.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass
                try:
                    process_pool.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass

            if config.email_send and counter_processed:
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                common.send_email(
                    subject=f"✅ Processing job finished on machine {config.machine_name}",
                    content=(
                        f"Processing job finished on {config.machine_name} at {time_now}. "
                        f"{counter_processed} segments were processed."
                    ),
                    sender=config.email_sender,
                    recipients=config.email_recipients,
                )

            dt_pass = time.time() - pass_start_ts
            logger.info(
                f"=== Pass {pass_index} completed === segments_processed={counter_processed} elapsed_sec={dt_pass:.2f}"
            )

            if config.sleep_sec and int(config.sleep_sec) > 0:
                helper.delete_youtube_mod_videos(video_paths)
                logger.info(f"Sleeping for {config.sleep_sec} s before attempting to go over mapping again.")
                time.sleep(config.sleep_sec)
                if config.git_pull:
                    common.git_pull()
                continue

            if config.git_pull:
                common.git_pull()

            # If no sleep configured, do one pass and exit.
            break

    except Exception as e:
        try:
            if "config" in locals() and getattr(config, "email_send", False):
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                common.send_email(
                    subject=f"‼️ Processing job crashed on machine {getattr(config, 'machine_name', 'unknown')}",
                    content=(
                        f"Processing job crashed on {getattr(config, 'machine_name', 'unknown')} at {time_now}. "
                        f"{counter_processed} segments were processed. Error message: {e}."
                    ),
                    sender=getattr(config, "email_sender", ""),
                    recipients=getattr(config, "email_recipients", []),
                )
        except Exception:
            pass
        raise
