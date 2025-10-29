# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
# -----------------------------------------------------------------------------
# Pipeline overview:
# - Reads a mapping CSV describing video IDs and segment intervals; runs in a loop.
# - Optionally updates metadata (ISO3, population, mortality, Gini, upload dates).
# - Retrieves videos (FTP first → YouTube fallback). With external_ssd=True:
#     * downloads go to videos[-1], and
#     * if a file already exists in videos[-1] and FTP is available, we refresh it:
#         download to a temp folder → delete old file → replace with fresh copy.
# - For each segment, run bbox and/or segmentation only if the corresponding CSV
#   is missing; otherwise skip work. If both modes are False, still download mp4s.
# - Cleans temp files and emails a summary or crash report.
# -----------------------------------------------------------------------------

import shutil
import os
import math
import glob  # checks for "do we already have CSVs?" without knowing FPS
from datetime import datetime
from helper_script import Youtube_Helper         # download/trim/track utilities
import pandas as pd
from custom_logger import CustomLogger           # structured logging
from logmod import logs                          # log level/color setup
import ast                                       # safe string->python list parsing
import common                                    # configs, secrets, email, git utils
from tqdm import tqdm                            # progress bar for mapping loop
import time                                      # sleep between passes
from types import SimpleNamespace                # lightweight config container

# Configure logging based on config file (verbosity & ANSI colors)
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)
helper = Youtube_Helper()

# Guard to avoid duplicate crash emails (kept for compatibility with some runners)
email_already_sent = False

# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    try:
        # ---------------------------------------------------------------------
        # Load static configuration (paths, switches) once
        # ---------------------------------------------------------------------
        config = SimpleNamespace(
            mapping=common.get_configs("mapping"),                 # path to mapping CSV
            videos=common.get_configs("videos"),                   # ordered list of video dirs
            delete_runs_files=common.get_configs("delete_runs_files"),
            delete_youtube_video=common.get_configs("delete_youtube_video"),
            data=common.get_configs("data"),                       # ordered list of data dirs
            countries_analyse=common.get_configs("countries_analyse"),
            update_ISO_code=common.get_configs("update_ISO_code"),
            update_pop_country=common.get_configs("update_pop_country"),
            update_mortality_rate=common.get_configs("update_mortality_rate"),
            update_gini_value=common.get_configs("update_gini_value"),
            update_upload_date=common.get_configs("update_upload_date"),
            segmentation_mode=common.get_configs("segmentation_mode"),
            tracking_mode=common.get_configs("tracking_mode"),
            save_annotated_video=common.get_configs("save_annotated_video"),
            sleep_sec=common.get_configs("sleep_sec"),
            git_pull=common.get_configs("git_pull"),
            machine_name=common.get_configs("machine_name"),
            email_send=common.get_configs("email_send"),
            email_sender=common.get_configs("email_sender"),
            email_recipients=common.get_configs("email_recipients"),
            compress_youtube_video=common.get_configs("compress_youtube_video"),
            external_ssd=common.get_configs("external_ssd"),       # True => prefer SSD target
            ftp_server=common.get_configs("ftp_server"),
        )

        # ---------------------------------------------------------------------
        # Load secrets (email + FTP credentials)
        # ---------------------------------------------------------------------
        secret = SimpleNamespace(
            email_smtp=common.get_secrets("email_smtp"),
            email_account=common.get_secrets("email_account"),
            email_password=common.get_secrets("email_password"),
            ftp_username=common.get_secrets("ftp_username"),
            ftp_password=common.get_secrets("ftp_password"),
        )

        # =========================================================================
        # Endless loop: process mapping; sleep; repeat
        # =========================================================================
        while True:
            # Read mapping each pass to pick up edits
            mapping = pd.read_csv(config.mapping)
            video_paths = config.videos  # convenience alias

            # Decide where to write downloads
            if config.external_ssd:
                # With SSD, ensure mount exists and set output to videos[-1]
                internal_ssd = config.videos[-1]
                os.makedirs(internal_ssd, exist_ok=True)
                output_path = config.videos[-1]
            else:
                output_path = config.videos[-1]

            # Convenience locals; keep behavior intact
            delete_runs_files = config.delete_runs_files
            delete_youtube_video = config.delete_youtube_video
            data_folders = config.data
            data_path = config.data[-1]  # final data root (where bbox/seg CSVs land)
            countries_analyse = config.countries_analyse
            counter_processed = 0        # segments processed this pass

            # -----------------------------------------------------------------
            # Optional mapping maintenance (all in-place on `mapping`)
            # -----------------------------------------------------------------
            if config.update_ISO_code:
                if "country" not in mapping.columns:
                    raise KeyError("The CSV file does not have a 'country' column.")
                if "iso3" not in mapping.columns:
                    mapping["iso3"] = None
                for index, row in mapping.iterrows():
                    mapping.at[index, "iso3"] = helper.get_iso_alpha_3(row["country"], row["iso3"])  # type: ignore
                mapping.to_csv(config.mapping, index=False)
                logger.info("Mapping file updated with ISO codes.")

            if config.update_pop_country:
                helper.update_population_in_csv(mapping)
            if config.update_mortality_rate:
                helper.fill_traffic_mortality(mapping)
            if config.update_gini_value:
                helper.fill_gini_data(mapping)

            if config.update_upload_date:
                # Compute upload dates for entries like "[id1,id2]"
                def extract_upload_dates(video_column):
                    upload_dates = []
                    for video_list in video_column:
                        video_ids = [vid.strip() for vid in video_list.strip('[]').split(',')]
                        dates = [helper.get_upload_date(vid) for vid in video_ids]
                        upload_dates.append(f"[{','.join(date if date else 'None' for date in dates)}]")
                    return upload_dates

                mapping['upload_date'] = extract_upload_dates(mapping['videos'])
                mapping.to_csv(config.mapping, index=False)
                logger.info("Mapping file updated successfully with upload dates.")

            # -----------------------------------------------------------------
            # Workspace setup: clean YOLO runs/* and ensure data dirs exist
            # -----------------------------------------------------------------
            helper.delete_folder(folder_path="runs")  # idempotent cleanup
            os.makedirs(data_path, exist_ok=True)
            os.makedirs(os.path.join(data_path, "bbox"), exist_ok=True)
            os.makedirs(os.path.join(data_path, "seg"), exist_ok=True)

            # =========================================================================
            # Iterate mapping rows; each may define multiple videos & segment lists
            # =========================================================================
            for index, row in tqdm(mapping.iterrows(), total=mapping.shape[0]):
                video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
                start_times = ast.literal_eval(row["start_time"])   # list[list[int]]
                end_times = ast.literal_eval(row["end_time"])       # list[list[int]]
                time_of_day = ast.literal_eval(row["time_of_day"])  # list[list[str]] (not used below)
                iso3 = str(row["iso3"])

                # Optional per-country filter (if provided)
                if countries_analyse and iso3 not in countries_analyse:
                    continue

                # Contextual log for monitoring
                city = str(row["city"])
                state = str(row["state"])
                country = str(row["country"])
                logger.info(f"Processing videos for city={city}, state={state}, country={country}.")

                # ---------------------------------------------------------------------
                # Each row: multiple video IDs, each with lists of segment starts/ends
                # ---------------------------------------------------------------------
                for vid_index, (vid, start_times_list, end_times_list, time_of_day_list) in enumerate(
                    zip(video_ids, start_times, end_times, time_of_day)
                ):
                    seg_mode_cfg = config.segmentation_mode
                    bbox_mode_cfg = config.tracking_mode

                    base_video_path = os.path.join(output_path, f"{vid}.mp4")  # presumed location after download
                    processed_flag = False
                    ftp_download = False
                    resolution = "unknown"  # may be set by downloader; used in archive naming

                    # =================================================================
                    # PRE-CHECK: Do we need this video at all?
                    # If both modes are False → ALWAYS download (archival).
                    # If any mode is True → skip download only if ALL required CSVs exist.
                    # Uses wildcard on FPS: {vid}_{start}_*.csv.
                    # =================================================================
                    video_needs_any_work = False
                    if seg_mode_cfg or bbox_mode_cfg:
                        for st, et in zip(start_times_list, end_times_list):
                            has_bbox = any(glob.glob(os.path.join(folder, "bbox", f"{vid}_{st}_*.csv"))
                                           for folder in data_folders)
                            has_seg = any(glob.glob(os.path.join(folder, "seg",  f"{vid}_{st}_*.csv"))
                                          for folder in data_folders)
                            missing_bbox = bool(bbox_mode_cfg and not has_bbox)
                            missing_seg = bool(seg_mode_cfg and not has_seg)
                            if missing_bbox or missing_seg:
                                video_needs_any_work = True
                                break
                        if not video_needs_any_work:
                            logger.info(f"{vid}: all required CSVs exist; skipping video (no download).")
                            continue
                    else:
                        # tracking_mode=False AND segmentation_mode=False → download-only run
                        logger.info(f"{vid}: modes disabled; downloading video for archival purposes.")

                    # =================================================================
                    # VIDEO RETRIEVAL: SSD-aware, FTP-first, refresh-if-exists logic
                    # =================================================================
                    if config.external_ssd:
                        existing_path = os.path.join(output_path, f"{vid}.mp4")
                        if os.path.exists(existing_path):
                            # ----------------------------------------------------------
                            # File already on SSD. Try to refresh from FTP SAFELY:
                            # 1) Download to a temp dir.
                            # 2) If FTP succeeds, delete the old file and replace.
                            # 3) If FTP not available, keep the old file.
                            # ----------------------------------------------------------
                            tmp_dir = os.path.join(output_path, "__tmp_dl")
                            os.makedirs(tmp_dir, exist_ok=True)

                            tmp_result = helper.download_videos_from_ftp(
                                filename=vid,
                                base_url=config.ftp_server,
                                out_dir=tmp_dir,                 # download to temp folder
                                username=secret.ftp_username,
                                password=secret.ftp_password,
                            )

                            if tmp_result:
                                # FTP had the file → replace the existing one
                                tmp_video_path, video_title, resolution, video_fps = tmp_result
                                if video_fps is None or video_fps == 0 or (isinstance(video_fps, float) and math.isnan(video_fps)):  # noqa: E501
                                    logger.warning("Invalid video_fps in refreshed file!")
                                    # cleanup temp and keep old file
                                    try:
                                        os.remove(tmp_video_path)
                                    except Exception:
                                        pass
                                else:
                                    # Remove old, move new into place
                                    try:
                                        os.remove(existing_path)
                                    except FileNotFoundError:
                                        pass
                                    final_path = os.path.join(output_path, f"{vid}.mp4")
                                    shutil.move(tmp_video_path, final_path)
                                    base_video_path = final_path
                                    ftp_download = True
                                    logger.info(f"{vid}: refreshed from FTP at {final_path}.")
                                    helper.set_video_title(video_title)
                                    if config.compress_youtube_video:
                                        helper.compress_video(base_video_path)
                            elif tmp_result is None:
                                # FTP not available → keep existing file; still allow YT fallback if needed below
                                logger.info(f"{vid}: FTP not available; keeping existing SSD copy.")
                                base_video_path = existing_path
                                video_title = vid
                                helper.set_video_title(video_title)
                                video_fps = helper.get_video_fps(base_video_path)
                                if video_fps is None or video_fps == 0 or (isinstance(video_fps, float) and math.isnan(video_fps)):  # noqa: E501
                                    logger.warning("Invalid video_fps on existing SSD copy; attempting YouTube fallback.")  # noqa: E501
                                    # Try YouTube fallback to refresh a bad file
                                    yt_result = helper.download_video_with_resolution(vid=vid, output_path=output_path)
                                    if yt_result:
                                        video_file_path, video_title, resolution, video_fps = yt_result
                                        if video_fps and (not isinstance(video_fps, float) or not math.isnan(video_fps)):  # noqa: E501
                                            base_video_path = video_file_path
                                            helper.set_video_title(video_title)
                                            if config.compress_youtube_video:
                                                helper.compress_video(base_video_path)
                                    else:
                                        logger.error(f"{vid}: YouTube fallback failed. Using existing (possibly bad) file.")  # noqa: E501
                        else:
                            # No SSD copy yet: do the usual FTP→YouTube download into videos[-1]
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
                                result = helper.download_video_with_resolution(vid=vid, output_path=output_path)

                            if result:
                                video_file_path, video_title, resolution, video_fps = result
                                if video_fps is None or video_fps == 0 or (isinstance(video_fps, float) and math.isnan(video_fps)):  # noqa: E501
                                    logger.warning("Invalid video_fps!")
                                    continue
                                base_video_path = video_file_path
                                logger.info(f"{vid}: downloaded to {video_file_path}.")
                                helper.set_video_title(video_title)
                                if config.compress_youtube_video:
                                    helper.compress_video(base_video_path)
                            else:
                                logger.error(f"{vid}: forced download failed (FTP+fallback). Skipping.")
                                continue

                    else:
                        # Default path: use caches from any video_paths; download only if missing
                        exists_somewhere = any(os.path.exists(os.path.join(path, f"{vid}.mp4")) for path in video_paths)  # noqa: E501
                        if not exists_somewhere:
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
                                result = helper.download_video_with_resolution(vid=vid, output_path=output_path)

                            if result:
                                video_file_path, video_title, resolution, video_fps = result
                                if video_fps is None or video_fps == 0 or (isinstance(video_fps, float) and math.isnan(video_fps)):  # noqa: E501
                                    logger.warning("Invalid video_fps!")
                                    continue
                                base_video_path = video_file_path
                                logger.info(f"{vid}: downloaded to {video_file_path}.")
                                helper.set_video_title(video_title)
                                if config.compress_youtube_video:
                                    helper.compress_video(base_video_path)
                            else:
                                # Last-resort: maybe file already exists at output_path
                                if os.path.exists(base_video_path):
                                    video_title = vid
                                    logger.info(f"{vid}: download failed, but video found locally at {base_video_path}.")  # noqa: E501
                                    helper.set_video_title(video_title)
                                    video_fps = helper.get_video_fps(base_video_path)
                                    if video_fps is None or video_fps == 0 or (isinstance(video_fps, float) and math.isnan(video_fps)):  # noqa: E501
                                        logger.warning("Invalid video_fps!")
                                        continue
                                else:
                                    logger.error(f"{vid}: video not found and download failed. Skipping.")
                                    continue
                        else:
                            # Use the first path that contains the file
                            logger.info(f"{vid}: using already downloaded video.")
                            existing_folder = next(
                                (path for path in video_paths if os.path.exists(os.path.join(path, f"{vid}.mp4"))),
                                None
                            )
                            existing_path = existing_folder if existing_folder else video_paths[-1]
                            base_video_path = os.path.join(existing_path, f"{vid}.mp4")
                            video_title = vid
                            helper.set_video_title(video_title)
                            video_fps = helper.get_video_fps(base_video_path)
                            if video_fps is None or video_fps == 0 or (isinstance(video_fps, float) and math.isnan(video_fps)):  # noqa: E501
                                logger.warning("Invalid video_fps!")
                                continue

                    # If both modes are disabled, this was a download-only pass
                    if not seg_mode_cfg and not bbox_mode_cfg:
                        continue

                    # =================================================================
                    # SEGMENT LOOP: Only run what's missing per segment (bbox/seg)
                    # =================================================================
                    for start_time, end_time, time_of_day_value in zip(start_times_list, end_times_list, time_of_day_list):  # noqa: E501
                        # Consistent FPS as integer for filenames
                        if video_fps is not None:
                            video_fps = int(video_fps)
                        segment_csv = f"{vid}_{start_time}_{video_fps}.csv"

                        # Check if outputs exist (strict, FPS-specific) across ALL data folders
                        has_bbox = any(os.path.isfile(os.path.join(folder, "bbox", segment_csv)) for folder in data_folders)  # noqa: E501
                        has_seg = any(os.path.isfile(os.path.join(folder, "seg",  segment_csv)) for folder in data_folders)  # noqa: E501

                        # Decide what to run per mode (independent)
                        run_bbox = bool(bbox_mode_cfg and not has_bbox)
                        run_seg = bool(seg_mode_cfg and not has_seg)

                        # If nothing needed for this segment, skip it early
                        if not run_bbox and not run_seg:
                            logger.info(f"{vid}: outputs exist for segment {start_time}; skipping segment.")
                            processed_flag = True
                            continue

                        # If SSD is used, ensure we operate on the SSD copy
                        if config.external_ssd:
                            try:
                                if os.path.dirname(base_video_path) != internal_ssd:
                                    out = helper.copy_video_safe(base_video_path, internal_ssd, vid)
                                    logger.debug(f"Copied to {out}.")
                            except Exception as exc:
                                # Add context for easier diagnosis
                                src_exists = os.path.isfile(base_video_path)
                                dest_file = os.path.join(internal_ssd, f"{vid}.mp4")
                                dest_parent_exists = os.path.isdir(internal_ssd)
                                logger.error(
                                    "[copy error] "
                                    f"src={base_video_path!r} src_exists={src_exists} "
                                    f"dest={dest_file!r} dest_parent_exists={dest_parent_exists} err={exc!r}"
                                )
                                raise
                            base_video_path = os.path.join(internal_ssd, f"{vid}.mp4")

                        # Temporary path for trimmed segment (on active drive)
                        trimmed_video_path = os.path.join(
                            internal_ssd if config.external_ssd else output_path,
                            f"{vid}_mod.mp4"
                        )

                        # Trim only if there's something to run for this segment
                        if start_time is None and end_time is None:
                            logger.info(f"{vid}: no trimming required for this video.")
                        elif run_bbox or run_seg:
                            logger.info(f"{vid}: trimming in progress for segment {start_time}-{end_time}s.")
                            end_time_adj = end_time - 1  # small guard against decoder boundary issues
                            helper.trim_video(base_video_path, trimmed_video_path, start_time, end_time_adj)
                            logger.info(f"{vid}: trimming completed for segment {start_time}-{end_time}s.")

                        # Run analysis (bbox/seg as needed)
                        if (run_bbox or run_seg) and video_fps > 0:  # type: ignore
                            logger.info(
                                f"{vid}: running analysis "
                                f"{'(bbox)' if run_bbox else ''}"
                                f"{' & ' if run_bbox and run_seg else ''}"
                                f"{'(seg)' if run_seg else ''} at {video_fps} FPS."
                            )
                            helper.tracking_mode(
                                trimmed_video_path,
                                output_path,
                                video_title=f"{vid}_mod.mp4",
                                video_fps=video_fps,
                                seg_mode=run_seg,               # run only needed parts
                                bbox_mode=run_bbox,
                                flag=config.save_annotated_video
                            )
                            counter_processed += 1
                            processed_flag = True
                        elif run_bbox or run_seg:
                            logger.warning(f"{vid}: invalid FPS ({video_fps}); skipping analysis.")

                        # Move/rename generated CSV(s) based on modes we actually ran
                        if run_bbox:
                            old = os.path.join("runs", "detect", f"{vid}.csv")
                            new = os.path.join("runs", "detect", segment_csv)
                            if os.path.exists(old):
                                os.rename(old, new)
                            if os.path.exists(new):
                                shutil.move(new, os.path.join(data_path, "bbox"))
                        if run_seg:
                            old = os.path.join("runs", "segment", f"{vid}.csv")
                            new = os.path.join("runs", "segment", segment_csv)
                            if os.path.exists(old):
                                os.rename(old, new)
                            if os.path.exists(new):
                                shutil.move(new, os.path.join(data_path, "seg"))

                        # Cleanup runs/* or archive, depending on config
                        if run_bbox or run_seg:
                            if delete_runs_files:
                                if run_bbox and os.path.isdir(os.path.join("runs", "detect")):
                                    shutil.rmtree(os.path.join("runs", "detect"))
                                if run_seg and os.path.isdir(os.path.join("runs", "segment")):
                                    shutil.rmtree(os.path.join("runs", "segment"))
                            else:
                                # Archive folders with a timestamp if keeping runs/*
                                ts = datetime.now()
                                if run_bbox and os.path.isdir(os.path.join("runs", "detect")):
                                    helper.rename_folder(os.path.join("runs", "detect"),
                                                         os.path.join("runs", f"{vid}_{resolution}_{ts}"))
                                if run_seg and os.path.isdir(os.path.join("runs", "segment")):
                                    helper.rename_folder(os.path.join("runs", "segment"),
                                                         os.path.join("runs", f"{vid}_{resolution}_{ts}"))

                        # Remove temporary trimmed file if tracking mode is enabled globally
                        if config.tracking_mode:
                            try:
                                os.remove(trimmed_video_path)
                            except FileNotFoundError:
                                pass

                    # -------------------------------------------------------------
                    # Per-video cleanup (base file) after all segments handled
                    # -------------------------------------------------------------
                    if config.external_ssd and processed_flag:
                        try:
                            os.remove(base_video_path)
                        except FileNotFoundError:
                            pass
                    if ftp_download:
                        try:
                            os.remove(base_video_path)
                        except FileNotFoundError:
                            pass
                    if delete_youtube_video:
                        try:
                            os.remove(os.path.join(output_path, f"{vid}.mp4"))
                        except FileNotFoundError:
                            pass

            # -----------------------------------------------------------------
            # Email success summary (only if anything was processed)
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
                    recipients=config.email_recipients
                )

            # -----------------------------------------------------------------
            # Sleep between passes; clean *_mod.mp4 remnants; optionally git pull
            # -----------------------------------------------------------------
            if config.sleep_sec:
                helper.delete_youtube_mod_videos(video_paths)
                logger.info(f"Sleeping for {config.sleep_sec} s before attempting to go over mapping again.")
                time.sleep(config.sleep_sec)

            if config.git_pull:
                common.git_pull()

    # =============================================================================
    # Crash handling: email details then re-raise for visibility
    # =============================================================================
    except Exception as e:
        try:
            if config.email_send:
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # kept from original (fun image): fine to send as plain link or ignore
                image_url = "https://i.pinimg.com/474x/20/82/0f/20820fd73c946d3e1d2e6efe23e1b2f3.jpg"
                common.send_email(
                    subject=f"‼️ Processing job crashed on machine {config.machine_name}",
                    content=(
                        f"Processing job crashed on {config.machine_name} at {time_now}. "
                        f"{counter_processed} segments were processed. Error message: {e}."
                    ),
                    sender=config.email_sender,
                    recipients=config.email_recipients
                )
        except Exception:
            # If config/email not ready or mail fails, swallow and re-raise
            pass
        raise
