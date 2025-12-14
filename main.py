# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
# -----------------------------------------------------------------------------
# Multithreaded rewrite (thread-safe CSV generation):
# - Same mapping loop + metadata update behavior.
# - Same video retrieval logic (FTP-first, SSD refresh, YouTube fallback).
# - Segment processing is concurrent via ThreadPoolExecutor.
# - Same model weights for every video (handled inside helper.tracking_mode_threadsafe()).
# - CSV outputs are written directly to data[-1]/bbox and data[-1]/seg
#   with the same naming and column schema as before.
# -----------------------------------------------------------------------------

import ast
import glob
import math
import os
import shutil
import threading
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from helper_script import Youtube_Helper
from logmod import logs


# Configure logging
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

# Helper (download/trim/etc.)
helper = Youtube_Helper()

# Guard to avoid duplicate crash emails (kept for compatibility with some runners)
email_already_sent = False

# Make tqdm updates thread-safe across worker threads
tqdm.set_lock(threading.RLock())


def _safe_get_config(key: str, default=None):
    try:
        return common.get_configs(key)
    except Exception:
        return default


def _ensure_dirs(data_path: str):
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.join(data_path, "bbox"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "seg"), exist_ok=True)


def _parse_bracket_list(s: str) -> list[str]:
    # mapping.csv uses strings like "[id1,id2]"
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    return [x.strip() for x in s.strip().strip("[]").split(",") if x.strip()]


def _has_any_outputs_wildcard(
    data_folders: list[str],
    vid: str,
    start_time: int,
    want_bbox: bool,
    want_seg: bool
) -> bool:
    """
    Wildcard existence check (FPS unknown): {vid}_{start}_*.csv
    Returns True if ALL required outputs exist (per requested modes).
    """
    has_bbox = True
    has_seg = True
    if want_bbox:
        has_bbox = any(
            glob.glob(os.path.join(folder, "bbox", f"{vid}_{start_time}_*.csv"))
            for folder in data_folders
        )
    if want_seg:
        has_seg = any(
            glob.glob(os.path.join(folder, "seg", f"{vid}_{start_time}_*.csv"))
            for folder in data_folders
        )
    return bool(has_bbox and has_seg)


def _outputs_exist_exact(data_folders: list[str], segment_csv: str) -> tuple[bool, bool]:
    has_bbox = any(os.path.isfile(os.path.join(folder, "bbox", segment_csv)) for folder in data_folders)
    has_seg = any(os.path.isfile(os.path.join(folder, "seg", segment_csv)) for folder in data_folders)
    return has_bbox, has_seg


def _fps_is_bad(fps) -> bool:
    return fps is None or fps == 0 or (isinstance(fps, float) and math.isnan(fps))


def _log_run_banner(config: SimpleNamespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    logger.info("============================================================")
    logger.info("Pipeline run configuration")
    logger.info(f"  device={device} gpu={gpu_name}")
    logger.info(f"  cpu_count={os.cpu_count()} torch_num_threads={torch.get_num_threads()}")
    logger.info(f"  max_workers={config.max_workers} (ThreadPoolExecutor)")
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


def _ensure_video_available(
    vid: str,
    config: SimpleNamespace,
    secret: SimpleNamespace,
    output_path: str,
    video_paths: list[str],
) -> tuple[str, str, str, int, bool]:
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
            # refresh safely from FTP into temp
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
                    return existing_path, video_title, resolution, int(video_fps2), False

                # replace
                try:
                    os.remove(existing_path)
                except FileNotFoundError:
                    pass

                final_path = os.path.join(output_path, f"{vid}.mp4")
                shutil.move(tmp_video_path, final_path)
                ftp_download = True
                helper.set_video_title(video_title)

                if config.compress_youtube_video:
                    helper.compress_video(final_path)

                logger.info(f"{vid}: refreshed from FTP and replaced SSD copy at {final_path}.")
                return final_path, video_title, resolution, int(video_fps), ftp_download

            # FTP unavailable; keep existing
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
                if config.compress_youtube_video:
                    helper.compress_video(video_file_path)
                return video_file_path, video_title, resolution, int(video_fps), False

            return existing_path, video_title, resolution, int(video_fps), False

        # no SSD copy yet: FTP -> YouTube
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
        if config.compress_youtube_video:
            helper.compress_video(video_file_path)

        logger.info(f"{vid}: downloaded successfully. res={resolution} fps={int(video_fps)} path={video_file_path}")
        return video_file_path, video_title, resolution, int(video_fps), ftp_download

    # external_ssd=False: cache-aware
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
            if config.compress_youtube_video:
                helper.compress_video(video_file_path)
            logger.info(f"{vid}: downloaded successfully. res={resolution} fps={int(video_fps)} path={video_file_path}")
            return video_file_path, video_title, resolution, int(video_fps), ftp_download

        # last resort: local at output_path
        if os.path.exists(base_video_path):
            helper.set_video_title(video_title)
            video_fps = helper.get_video_fps(base_video_path)
            if _fps_is_bad(video_fps):
                raise RuntimeError(f"{vid}: invalid FPS on local fallback file.")
            logger.info(f"{vid}: found locally at {base_video_path}. fps={int(video_fps)}")
            return base_video_path, video_title, resolution, int(video_fps), False

        raise RuntimeError(f"{vid}: video not found and download failed.")

    # use first path that contains file
    existing_folder = next((p for p in video_paths if os.path.exists(os.path.join(p, f"{vid}.mp4"))), None)
    use_folder = existing_folder if existing_folder else video_paths[-1]
    base_video_path = os.path.join(use_folder, f"{vid}.mp4")
    helper.set_video_title(video_title)

    video_fps = helper.get_video_fps(base_video_path)
    if _fps_is_bad(video_fps):
        raise RuntimeError(f"{vid}: invalid FPS on cached file.")

    logger.info(f"{vid}: using cached video at {base_video_path}. fps={int(video_fps)}")
    return base_video_path, video_title, resolution, int(video_fps), False


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    counter_processed = 0  # ensure defined for crash email

    try:
        # ---------------------------------------------------------------------
        # Load static configuration once (use safe reads where possible)
        # ---------------------------------------------------------------------
        config = SimpleNamespace(
            mapping=common.get_configs("mapping"),
            videos=common.get_configs("videos"),
            data=common.get_configs("data"),
            countries_analyse=_safe_get_config("countries_analyse", []),

            # mapping maintenance flags (safe)
            update_pop_country=_safe_get_config("update_pop_country", False),
            update_gini_value=_safe_get_config("update_gini_value", False),

            # modes
            segmentation_mode=_safe_get_config("segmentation_mode", False),
            tracking_mode=_safe_get_config("tracking_mode", False),

            # housekeeping
            delete_youtube_video=_safe_get_config("delete_youtube_video", False),
            compress_youtube_video=_safe_get_config("compress_youtube_video", False),
            external_ssd=_safe_get_config("external_ssd", False),

            # server
            ftp_server=_safe_get_config("ftp_server", None),

            # models/trackers (same keys helper uses)
            tracking_model=_safe_get_config("tracking_model", ""),
            segment_model=_safe_get_config("segment_model", ""),
            bbox_tracker=_safe_get_config("bbox_tracker", ""),
            seg_tracker=_safe_get_config("seg_tracker", ""),
            track_buffer_sec=_safe_get_config("track_buffer_sec", 1),

            # runtime behavior
            save_annotated_video=_safe_get_config("save_annotated_video", False),
            sleep_sec=_safe_get_config("sleep_sec", 0),
            git_pull=_safe_get_config("git_pull", False),

            # email
            machine_name=_safe_get_config("machine_name", "unknown"),
            email_send=_safe_get_config("email_send", False),
            email_sender=_safe_get_config("email_sender", ""),
            email_recipients=_safe_get_config("email_recipients", []),

            # concurrency
            max_workers=int(_safe_get_config("max_workers", 2)),
        )

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

        # Normalize workers
        config.max_workers = max(1, int(config.max_workers))

        _log_run_banner(config)

        pass_index = 0

        while True:
            pass_index += 1
            pass_start_ts = time.time()

            mapping = pd.read_csv(config.mapping)
            video_paths = config.videos

            # Download destination
            if config.external_ssd:
                internal_ssd = config.videos[-1]
                os.makedirs(internal_ssd, exist_ok=True)
                output_path = config.videos[-1]
            else:
                internal_ssd = None
                output_path = config.videos[-1]

            data_folders = config.data
            data_path = config.data[-1]
            countries_analyse = config.countries_analyse or []
            counter_processed = 0

            logger.info(
                f"=== Pass {pass_index} started === rows={mapping.shape[0]} "
                f"max_workers={config.max_workers} active_threads_now={threading.active_count()} "
                f"tracking_mode={bool(config.tracking_mode)} seg_mode={bool(config.segmentation_mode)}"
            )

            # -----------------------------------------------------------------
            # Optional mapping maintenance
            # -----------------------------------------------------------------
            if config.update_pop_country:
                logger.info("Updating population in mapping file...")
                helper.update_population_in_csv(mapping)

            if config.update_gini_value:
                logger.info("Updating GINI values in mapping file...")
                helper.fill_gini_data(mapping)

            # -----------------------------------------------------------------
            # Workspace setup (runs/ cleanup kept for compatibility)
            # -----------------------------------------------------------------
            helper.delete_folder(folder_path="runs")
            _ensure_dirs(data_path)

            # =========================================================================
            # Iterate mapping rows
            # =========================================================================
            pbar_rows = tqdm(mapping.iterrows(), total=mapping.shape[0], desc="Mapping rows", dynamic_ncols=True)
            for _, row in pbar_rows:
                video_ids = _parse_bracket_list(str(row["videos"]))
                if not video_ids:
                    continue

                start_times = ast.literal_eval(row["start_time"])
                end_times = ast.literal_eval(row["end_time"])
                time_of_day = ast.literal_eval(row["time_of_day"])

                iso3 = str(row.get("iso3", ""))

                if countries_analyse and iso3 and iso3 not in countries_analyse:
                    continue

                city = str(row.get("city", ""))
                state = str(row.get("state", ""))
                country = str(row.get("country", ""))

                logger.info(f"Row context: city={city}, state={state}, country={country}, iso3={iso3}")

                for vid, start_times_list, end_times_list, time_of_day_list in zip(
                    video_ids, start_times, end_times, time_of_day
                ):
                    seg_mode_cfg = bool(config.segmentation_mode)
                    bbox_mode_cfg = bool(config.tracking_mode)

                    # =================================================================
                    # PRE-CHECK (wildcard FPS) to decide whether to download at all
                    # =================================================================
                    if seg_mode_cfg or bbox_mode_cfg:
                        needs_any_work = False
                        for st, et in zip(start_times_list, end_times_list):
                            if not _has_any_outputs_wildcard(data_folders, vid, int(st), bbox_mode_cfg, seg_mode_cfg):
                                needs_any_work = True
                                break
                        if not needs_any_work:
                            logger.info(f"{vid}: all required CSVs exist; skipping video (no download).")
                            continue
                    else:
                        logger.info(f"{vid}: modes disabled; downloading video for archival purposes.")

                    # =================================================================
                    # VIDEO RETRIEVAL
                    # =================================================================
                    t_vid0 = time.time()
                    base_video_path, video_title, resolution, video_fps, ftp_download = _ensure_video_available(
                        vid=vid,
                        config=config,
                        secret=secret,
                        output_path=output_path,
                        video_paths=video_paths,
                    )

                    logger.info(
                        f"{vid}: ready. fps={int(video_fps)} res={resolution} "
                        f"external_ssd={config.external_ssd} ftp_download={ftp_download} "
                        f"elapsed_sec={time.time() - t_vid0:.2f}"
                    )

                    # If both modes are disabled: download-only
                    if not seg_mode_cfg and not bbox_mode_cfg:
                        continue

                    # If SSD: ensure file is on SSD once per video
                    if config.external_ssd and internal_ssd:
                        base_video_path = _copy_to_ssd_if_needed(base_video_path, internal_ssd, vid)

                    # =================================================================
                    # Build segment jobs (only missing outputs, FPS-specific)
                    # =================================================================
                    segment_jobs = []
                    for start_time, end_time, _tod in zip(start_times_list, end_times_list, time_of_day_list):
                        start_time = int(start_time)
                        end_time = int(end_time)

                        segment_csv = f"{vid}_{start_time}_{int(video_fps)}.csv"
                        has_bbox, has_seg = _outputs_exist_exact(data_folders, segment_csv)

                        run_bbox = bool(bbox_mode_cfg and not has_bbox)
                        run_seg = bool(seg_mode_cfg and not has_seg)

                        if not run_bbox and not run_seg:
                            logger.info(f"{vid}: outputs exist for segment {start_time}; skipping segment.")
                            continue

                        # Unique trimmed filename per segment to be thread-safe
                        work_dir = internal_ssd if (config.external_ssd and internal_ssd) else output_path
                        trimmed_video_path = os.path.join(work_dir, f"{vid}_{start_time}_{end_time}_mod.mp4")

                        bbox_csv_out = os.path.join(data_path, "bbox", segment_csv) if run_bbox else None
                        seg_csv_out = os.path.join(data_path, "seg", segment_csv) if run_seg else None

                        segment_jobs.append(
                            (start_time, end_time, trimmed_video_path, run_bbox, run_seg, bbox_csv_out, seg_csv_out)
                        )

                    if not segment_jobs:
                        continue

                    logger.info(
                        f"{vid}: scheduling {len(segment_jobs)} segment(s) "
                        f"max_workers={config.max_workers} active_threads_now={threading.active_count()}"
                    )

                    # =================================================================
                    # MULTITHREADED EXECUTION (per-segment)
                    # =================================================================
                    processed_any_for_video = False

                    def _segment_worker(job):
                        st, et, trimmed_path, do_bbox, do_seg, bbox_out, seg_out = job
                        th = threading.current_thread().name
                        t0 = time.time()

                        mode_str = f"{'bbox' if do_bbox else ''}{'&' if do_bbox and do_seg else ''}{'seg' if do_seg else ''}"
                        job_label = f"{vid} [{st}-{et}s] ({mode_str})"

                        # Map each worker thread to a fixed tqdm line position:
                        # position=0 is typically used by your per-video segment bar, so use 1..max_workers for workers.
                        try:
                            worker_idx = int(th.split("_")[-1])  # ThreadPoolExecutor-0_0 -> 0
                        except Exception:
                            worker_idx = 0
                        tqdm_pos = 1 + worker_idx

                        logger.info(
                            f"[worker-start] thread={th} job={job_label} fps={int(video_fps)} "
                            f"trimmed={os.path.basename(trimmed_path)}"
                        )

                        try:
                            # Track directly on the source video window (no trimming)
                            logger.info(f"[track-start] thread={th} job={job_label} window={st}-{et}s")
                            helper.tracking_mode_threadsafe_window(
                                source_video_path=base_video_path,
                                start_time_s=st,
                                end_time_s=et,
                                video_fps=int(video_fps),
                                bbox_mode=do_bbox,
                                seg_mode=do_seg,
                                bbox_csv_out=bbox_out,
                                seg_csv_out=seg_out,
                                job_label=job_label,
                                tqdm_position=tqdm_pos,
                                show_frame_pbar=True,
                                postfix_every_n=30,
                            )
                            logger.info(f"[track-done] thread={th} job={job_label}")

                            if bbox_out and os.path.exists(bbox_out):
                                logger.info(f"[csv] thread={th} job={job_label} bbox_bytes={os.path.getsize(bbox_out)}")
                            if seg_out and os.path.exists(seg_out):
                                logger.info(f"[csv] thread={th} job={job_label} seg_bytes={os.path.getsize(seg_out)}")

                            return 1

                        finally:
                            # try:
                            #     os.remove(trimmed_path)
                            # except FileNotFoundError:
                            #     pass

                            dt = time.time() - t0
                            logger.info(f"[worker-done] thread={th} job={job_label} elapsed_sec={dt:.2f}")

                    # Execute futures + show per-video progress bar
                    with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
                        futures = [ex.submit(_segment_worker, job) for job in segment_jobs]

                        with tqdm(
                            total=len(futures),
                            desc=f"{vid} segments",
                            unit="seg",
                            dynamic_ncols=True,
                            leave=False,
                        ) as seg_pbar:
                            for f in as_completed(futures):
                                counter_processed += f.result()
                                processed_any_for_video = True
                                seg_pbar.update(1)

                    # =================================================================
                    # Per-video cleanup
                    # =================================================================
                    if config.external_ssd and processed_any_for_video:
                        try:
                            os.remove(base_video_path)
                        except FileNotFoundError:
                            pass

                    if ftp_download:
                        try:
                            os.remove(base_video_path)
                        except FileNotFoundError:
                            pass

                    if config.delete_youtube_video:
                        try:
                            os.remove(os.path.join(output_path, f"{vid}.mp4"))
                        except FileNotFoundError:
                            pass

            pbar_rows.close()

            # -----------------------------------------------------------------
            # Email success summary
            # -----------------------------------------------------------------
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
            logger.info(f"=== Pass {pass_index} completed === segments_processed={counter_processed} elapsed_sec={dt_pass:.2f}")

            # -----------------------------------------------------------------
            # Sleep + housekeeping + optional git pull
            # -----------------------------------------------------------------
            if config.sleep_sec:
                helper.delete_youtube_mod_videos(video_paths)
                logger.info(f"Sleeping for {config.sleep_sec} s before attempting to go over mapping again.")
                time.sleep(config.sleep_sec)

            if config.git_pull:
                common.git_pull()

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
