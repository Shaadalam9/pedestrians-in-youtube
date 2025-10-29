# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
# -----------------------------------------------------------------------------
# Script overview:
# - Continuously reads a mapping CSV describing YouTube/FTP video IDs and segments.
# - Ensures metadata columns (ISO codes, population, mortality, Gini, upload date).
# - Retrieves videos (FTP preferred; YouTube fallback), trims segments, runs detection/segmentation,
#   and saves CSV outputs into data folders (bbox/seg).
# - When external_ssd=True, ALWAYS download into videos[-1] (SSD) and ignore caches.
# - Sends a summary email after each run and a crash email on exceptions.
# -----------------------------------------------------------------------------

import shutil
import os
import math
from datetime import datetime
from helper_script import Youtube_Helper  # Helper with downloading, trimming, tracking utils
import pandas as pd
from custom_logger import CustomLogger      # Structured/colored logs
from logmod import logs                     # Logging level/color configurator
import ast                                   # Safely parse string lists (from CSV) into Python objects
import common                                # Shared configs/secrets/email/git utilities
from tqdm import tqdm                        # CLI progress bars for long loops
import time                                  # Sleep between runs
from types import SimpleNamespace            # Lightweight config container

# Initialize logging system (level & color from config)
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)
helper = Youtube_Helper()

# Flag used to guard against spamming crash emails (kept from original structure)
email_already_sent = False

# =============================================================================
# Main execution guard: run continuous pipeline until interrupted
# =============================================================================
if __name__ == "__main__":
    # Wrap the entire loop to catch and notify on crashes
    try:
        # ---------------------------------------------------------------------
        # Load static configuration values once (paths, switches, booleans)
        # ---------------------------------------------------------------------
        config = SimpleNamespace(
            mapping=common.get_configs("mapping"),                 # Path to mapping CSV
            videos=common.get_configs("videos"),                   # List of video directories (ordered)
            delete_runs_files=common.get_configs("delete_runs_files"),  # Whether to delete YOLO runs/*
            delete_youtube_video=common.get_configs("delete_youtube_video"),  # Remove original mp4 post-processing
            data=common.get_configs("data"),                       # List of data directories (ordered)
            countries_analyse=common.get_configs("countries_analyse"),  # Optional ISO3 filter
            update_ISO_code=common.get_configs("update_ISO_code"),       # Maintenance flags:
            update_pop_country=common.get_configs("update_pop_country"),
            update_mortality_rate=common.get_configs("update_mortality_rate"),
            update_gini_value=common.get_configs("update_gini_value"),
            update_upload_date=common.get_configs("update_upload_date"),
            segmentation_mode=common.get_configs("segmentation_mode"),    # Segmentation toggle
            tracking_mode=common.get_configs("tracking_mode"),            # Detection (bbox) toggle
            save_annotated_video=common.get_configs("save_annotated_video"),  # Save annotated video or not
            sleep_sec=common.get_configs("sleep_sec"),                  # Delay between full passes
            git_pull=common.get_configs("git_pull"),                    # Pull latest mapping from git
            machine_name=common.get_configs("machine_name"),            # For email subjects
            email_send=common.get_configs("email_send"),                # Email notifications toggle
            email_sender=common.get_configs("email_sender"),            # Sender address
            email_recipients=common.get_configs("email_recipients"),    # Recipient list
            compress_youtube_video=common.get_configs("compress_youtube_video"),  # Optional compression
            external_ssd=common.get_configs("external_ssd"),         # If True: force downloads to videos[-1]
            ftp_server=common.get_configs("ftp_server"),             # FTP host/base URL
        )

        # ---------------------------------------------------------------------
        # Load secrets once (email + FTP)
        # ---------------------------------------------------------------------
        secret = SimpleNamespace(
            email_smtp=common.get_secrets("email_smtp"),            # SMTP host
            email_account=common.get_secrets("email_account"),      # SMTP username
            email_password=common.get_secrets("email_password"),    # SMTP password
            ftp_username=common.get_secrets("ftp_username"),        # FTP username
            ftp_password=common.get_secrets("ftp_password"),        # FTP password
        )

        # =========================================================================
        # Outer loop: continuously process mapping file (intended to run forever)
        # =========================================================================
        while True:
            # ---------------------------------------------------------------------
            # Read mapping (must contain columns: videos, start_time, end_time, etc.)
            # ---------------------------------------------------------------------
            mapping = pd.read_csv(config.mapping)

            # Ordered list of folders where videos may reside; last one is default target
            video_paths = config.videos

            # ---------------------------------------------------------------------
            # Decide output path(s) based on external_ssd flag
            # ---------------------------------------------------------------------
            if config.external_ssd:
                # With external SSD, ensure mount exists and set output to SSD folder (videos[-1])
                internal_ssd = config.videos[-1]
                os.makedirs(internal_ssd, exist_ok=True)
                output_path = config.videos[-1]     # IMPORTANT: save downloads in videos[-1]
            else:
                # Default: also use the last configured videos folder
                output_path = config.videos[-1]

            # Convenience locals (unchanged behavior)
            delete_runs_files = config.delete_runs_files
            delete_youtube_video = config.delete_youtube_video
            data_folders = config.data
            data_path = config.data[-1]             # Where CSV outputs will be placed
            countries_analyse = config.countries_analyse
            counter_processed = 0                   # Count of processed segments in this pass

            # ---------------------------------------------------------------------
            # Optional mapping maintenance steps (enabled via config flags)
            # ---------------------------------------------------------------------
            if config.update_ISO_code:
                # Fail early if mapping lacks country column
                if "country" not in mapping.columns:
                    raise KeyError("The CSV file does not have a 'country' column.")
                # Create iso3 column if missing
                if "iso3" not in mapping.columns:
                    mapping["iso3"] = None

                # Update ISO alpha-3 for each row (using helper that respects existing value when valid)
                for index, row in mapping.iterrows():
                    mapping.at[index, "iso3"] = helper.get_iso_alpha_3(row["country"], row["iso3"])  # type: ignore

                mapping.to_csv(config.mapping, index=False)
                logger.info("Mapping file updated with ISO codes.")

            # These helpers enrich mapping with population, mortality, and Gini data (in-place)
            if config.update_pop_country:
                helper.update_population_in_csv(mapping)
            if config.update_mortality_rate:
                helper.fill_traffic_mortality(mapping)
            if config.update_gini_value:
                helper.fill_gini_data(mapping)

            # Update upload_date for each list of YouTube IDs (stringified list in CSV)
            if config.update_upload_date:
                def extract_upload_dates(video_column):
                    """Parse 'videos' col entries like '[id1,id2]' and fetch upload dates for each."""
                    upload_dates = []
                    for video_list in video_column:
                        # Parse delimited list without using ast (consistent with original style)
                        video_ids = video_list.strip('[]').split(',')
                        video_ids = [vid.strip() for vid in video_ids]
                        # Lookup each video's upload date (returns ISO string or None)
                        dates = [helper.get_upload_date(vid) for vid in video_ids]
                        # Persist back as a bracketed comma-joined list to match input format
                        upload_dates.append(f"[{','.join(date if date else 'None' for date in dates)}]")
                    return upload_dates

                mapping['upload_date'] = extract_upload_dates(mapping['videos'])
                mapping.to_csv(config.mapping, index=False)
                logger.info("Mapping file updated successfully with upload dates.")

            # ---------------------------------------------------------------------
            # Workspace prep: clean runs/* and ensure data subfolders exist
            # ---------------------------------------------------------------------
            helper.delete_folder(folder_path="runs")                               # idempotent cleanup
            os.makedirs(data_path, exist_ok=True)                                  # main data path
            os.makedirs(os.path.join(data_path, "bbox"), exist_ok=True)            # bbox CSVs
            os.makedirs(os.path.join(data_path, "seg"), exist_ok=True)             # seg CSVs

            # =========================================================================
            # Iterate over mapping rows (each row can include multiple video IDs & segments)
            # =========================================================================
            for index, row in tqdm(mapping.iterrows(), total=mapping.shape[0]):
                # Parse video IDs as strings from a bracketed list like "[id1,id2]"
                video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]

                # Parse segment lists (lists of lists) safely using ast.literal_eval
                start_times = ast.literal_eval(row["start_time"])     # e.g., [[0, 30], [10, 40]]
                end_times = ast.literal_eval(row["end_time"])
                time_of_day = ast.literal_eval(row["time_of_day"])    # optional metadata (unused in logic below)
                iso3 = str(row["iso3"])                               # country filter (if configured)

                # Optional filter: process only selected countries (if countries_analyse is non-empty)
                if countries_analyse and iso3 not in countries_analyse:
                    continue

                # Log contextual info for observability
                city = str(row["city"])
                state = str(row["state"])
                country = str(row["country"])
                logger.info(f"Processing videos for city={city}, state={state}, country={country}.")

                # ---------------------------------------------------------------------
                # Each row can specify multiple independent (video_id, segments) bundles
                # ---------------------------------------------------------------------
                for vid_index, (vid, start_times_list, end_times_list, time_of_day_list) in enumerate(
                    zip(video_ids, start_times, end_times, time_of_day)
                ):
                    # Mode toggles are read per-video (unchanged behavior)
                    seg_mode = config.segmentation_mode
                    bbox_mode = config.tracking_mode

                    # Default assumption: if we fetch, it lands at output_path/vid.mp4
                    base_video_path = os.path.join(output_path, f"{vid}.mp4")
                    processed_flag = False   # Whether any segment for this vid was processed
                    ftp_download = False     # To decide cleanup of FTP-downloaded file at the end
                    resolution = "unknown"   # Named only for logging/renames; may be set by downloader

                    # =================================================================
                    # VIDEO RETRIEVAL LOGIC
                    # =================================================================
                    if config.external_ssd:
                        # ------------------------------------------------------------
                        # FORCE download when on SSD:
                        # - First try FTP to output_path (videos[-1])
                        # - If FTP result is None (not error, but "not found"), fallback to YouTube
                        # - Ignore any previously cached copies in video_paths
                        # ------------------------------------------------------------
                        result = helper.download_videos_from_ftp(
                            filename=vid,
                            base_url=config.ftp_server,
                            out_dir=output_path,             # IMPORTANT: videos[-1]
                            username=secret.ftp_username,
                            password=secret.ftp_password,
                        )
                        if result:
                            ftp_download = True              # Remember we pulled from FTP for later cleanup
                        if result is None:
                            # If FTP couldn't provide the file, try YouTube download with resolution detection
                            result = helper.download_video_with_resolution(vid=vid, output_path=output_path)

                        if result:
                            # Unpack downloader tuple: (file_path, title, resolution, fps)
                            video_file_path, video_title, resolution, video_fps = result

                            # Validate FPS to avoid downstream errors
                            if video_fps is None or video_fps == 0 or (isinstance(video_fps,
                                                                                  float) and math.isnan(video_fps)):
                                logger.warning("Invalid video_fps!")
                                continue

                            # Point base_video_path to freshly downloaded file
                            base_video_path = video_file_path
                            logger.info(f"{vid}: downloaded to {video_file_path}.")
                            helper.set_video_title(video_title)

                            # Optionally compress original video to save space/bandwidth
                            if config.compress_youtube_video:
                                helper.compress_video(base_video_path)
                        else:
                            # Both FTP and YouTube failed → skip this video
                            logger.error(f"{vid}: forced download failed (FTP+fallback). Skipping.")
                            continue

                    else:
                        # ------------------------------------------------------------
                        # DEFAULT MODE: Prefer caches (video_paths) and download only if missing
                        # ------------------------------------------------------------
                        exists_somewhere = any(os.path.exists(os.path.join(path,
                                                                           f"{vid}.mp4")) for path in video_paths)

                        if not exists_somewhere:
                            # Attempt FTP download into output_path (videos[-1])
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
                                # Fallback to YouTube if FTP returns no file
                                result = helper.download_video_with_resolution(vid=vid, output_path=output_path)

                            if result:
                                video_file_path, video_title, resolution, video_fps = result
                                if video_fps is None or video_fps == 0 or (isinstance(video_fps, float) and math.isnan(video_fps)):  # noqa:E501
                                    logger.warning("Invalid video_fps!")
                                    continue

                                base_video_path = video_file_path
                                logger.info(f"{vid}: downloaded to {video_file_path}.")
                                helper.set_video_title(video_title)
                                if config.compress_youtube_video:
                                    helper.compress_video(base_video_path)
                            else:
                                # As a last resort, check if the presumed output file exists already
                                if os.path.exists(base_video_path):
                                    video_title = vid                  # Fallback title if unknown
                                    logger.info(f"{vid}: download failed, but video found locally at {base_video_path}.")  # noqa:E501
                                    helper.set_video_title(video_title)
                                    video_fps = helper.get_video_fps(base_video_path)
                                    if video_fps is None or video_fps == 0 or (isinstance(video_fps, float) and math.isnan(video_fps)):  # noqa:E501
                                        logger.warning("Invalid video_fps!")
                                        continue
                                else:
                                    logger.error(f"{vid}: video not found and download failed. Skipping.")
                                    continue
                        else:
                            # Use the first folder that contains the file (preserve original behavior)
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
                            if video_fps is None or video_fps == 0 or (isinstance(video_fps, float) and math.isnan(video_fps)):  # noqa:E501
                                logger.warning("Invalid video_fps!")
                                continue

                    # =================================================================
                    # PER-SEGMENT PROCESSING (trim + detection/seg + CSV export)
                    # =================================================================
                    for start_time, end_time, time_of_day_value in zip(start_times_list, end_times_list, time_of_day_list):  # noqa:E501
                        # Track file paths discovered across all configured data folders
                        bbox_paths, seg_paths = [], []

                        # Use integer FPS for consistent filenames and YOLO expectations
                        if video_fps is not None:
                            video_fps = int(video_fps)

                        # Output CSV basename pattern includes vid, segment start, and fps
                        filename = f"{vid}_{start_time}_{video_fps}.csv"

                        # Look for existing outputs (so we can skip reprocessing)
                        for folder in data_folders:
                            bbox_path = os.path.join(folder, "bbox", filename)
                            seg_path = os.path.join(folder, "seg", filename)
                            if os.path.isfile(bbox_path):
                                bbox_paths.append(bbox_path)
                            if os.path.isfile(seg_path):
                                seg_paths.append(seg_path)

                        # Decide which modes (bbox/seg) to run for this segment
                        found_path = None
                        bbox_mode = False
                        seg_mode = False

                        if config.tracking_mode and not config.segmentation_mode:
                            # Only bbox requested
                            if bbox_paths:
                                found_path = bbox_paths
                                continue     # Skip; already have bbox output
                            else:
                                bbox_mode = True
                        elif config.segmentation_mode and not config.tracking_mode:
                            # Only segmentation requested
                            if seg_paths:
                                found_path = seg_paths
                                continue     # Skip; already have seg output
                            else:
                                seg_mode = True
                        elif config.tracking_mode and config.segmentation_mode:
                            # Both requested: if both exist, skip; else run missing part(s)
                            if bbox_paths and seg_paths:
                                found_path = list(zip(bbox_paths, seg_paths))
                                continue
                            elif bbox_paths:
                                found_path = bbox_paths
                                seg_mode = True   # need seg
                            elif seg_paths:
                                found_path = seg_paths
                                bbox_mode = True  # need bbox
                            else:
                                bbox_mode = seg_mode = True
                        else:
                            # Neither mode enabled → nothing to do for this segment
                            found_path = None

                        # If neither mode is required (i.e., outputs exist), skip
                        if not bbox_mode and not seg_mode:
                            logger.info(f"{vid}: YOLO file {filename} exists. Skipping segment.")
                            processed_flag = True
                            continue

                        # ------------------------------------------------------------
                        # If external SSD is used, ensure processing uses SSD path
                        # ------------------------------------------------------------
                        if config.external_ssd:
                            try:
                                # Only copy if the file isn't already in the SSD folder
                                if os.path.dirname(base_video_path) != internal_ssd:
                                    out = helper.copy_video_safe(base_video_path, internal_ssd, vid)
                                    logger.debug(f"Copied to {out}.")
                            except Exception as exc:
                                # Log additional context for easier debugging
                                src_exists = os.path.isfile(base_video_path)
                                dest_file = os.path.join(internal_ssd, f"{vid}.mp4")
                                dest_parent_exists = os.path.isdir(internal_ssd)
                                logger.error(
                                    "[copy error] "
                                    f"src={base_video_path!r} src_exists={src_exists} "
                                    f"dest={dest_file!r} dest_parent_exists={dest_parent_exists} err={exc!r}"
                                )
                                raise
                            # Switch base to SSD copy for trimming and inference
                            base_video_path = os.path.join(internal_ssd, f"{vid}.mp4")

                        # Temporary trimmed video path (lives on SSD when applicable)
                        trimmed_video_path = os.path.join(
                            internal_ssd if config.external_ssd else output_path,
                            f"{video_title}_mod.mp4"
                        )

                        # Decide whether to trim (only when start/end provided and some mode is active)
                        if start_time is None and end_time is None:
                            logger.info(f"{vid}: no trimming required for this video.")
                        elif bbox_mode or seg_mode:
                            logger.info(f"{vid}: trimming in progress for segment {start_time}-{end_time}s.")
                            # Subtract 1 sec to avoid potential decoder boundary issues at the tail
                            end_time_adj = end_time - 1
                            helper.trim_video(base_video_path, trimmed_video_path, start_time, end_time_adj)
                            logger.info(f"{vid}: trimming completed for segment {start_time}-{end_time}s.")

                        # ------------------------------------------------------------
                        # Run detection/segmentation on the trimmed segment
                        # ------------------------------------------------------------
                        if seg_mode or bbox_mode:
                            if video_fps > 0:  # Guard against invalid FPS
                                logger.info(
                                    f"{vid}: YOLO analysis for {start_time}-{end_time}s at {video_fps} FPS."
                                )
                                helper.tracking_mode(
                                    trimmed_video_path,                    # input segment
                                    output_path,                           # where annotated output goes (if enabled)
                                    video_title=f"{video_title}_mod.mp4",  # propagated title for helper
                                    video_fps=video_fps,                   # ensures consistent frame stepping
                                    seg_mode=seg_mode,
                                    bbox_mode=bbox_mode,
                                    flag=config.save_annotated_video       # toggle saving annotated video
                                )
                                counter_processed += 1
                                processed_flag = True
                            else:
                                logger.warning(f"{vid}: FPS value is {video_fps}. Skipping tracking mode.")

                            # --------------------------------------------------------
                            # Move/rename produced CSV(s) from runs/* to data folders
                            # --------------------------------------------------------
                            if bbox_mode:
                                old_file_path = os.path.join("runs", "detect", f"{vid}.csv")
                                new_file_path = os.path.join("runs", "detect", f"{vid}_{start_time}_{video_fps}.csv")
                                if os.path.exists(old_file_path):
                                    os.rename(old_file_path, new_file_path)
                                else:
                                    logger.error(f"{vid}: error:{old_file_path} does not exist.")
                                if os.path.exists(new_file_path):
                                    shutil.move(new_file_path, os.path.join(data_path, "bbox"))
                                else:
                                    logger.error(f"{vid}: error: {new_file_path} does not exist.")

                            if seg_mode:
                                old_file_path = os.path.join("runs", "segment", f"{vid}.csv")
                                new_file_path = os.path.join("runs", "segment", f"{vid}_{start_time}_{video_fps}.csv")
                                if os.path.exists(old_file_path):
                                    os.rename(old_file_path, new_file_path)
                                else:
                                    logger.error(f"{vid}: error:{old_file_path} does not exist.")
                                if os.path.exists(new_file_path):
                                    shutil.move(new_file_path, os.path.join(data_path, "seg"))
                                else:
                                    logger.error(f"{vid}: error: {new_file_path} does not exist.")

                            # --------------------------------------------------------
                            # Cleanup or archive runs/* depending on delete_runs_files
                            # --------------------------------------------------------
                            if delete_runs_files:
                                if bbox_mode and os.path.isdir(os.path.join("runs", "detect")):
                                    shutil.rmtree(os.path.join("runs", "detect"))
                                if seg_mode and os.path.isdir(os.path.join("runs", "segment")):
                                    shutil.rmtree(os.path.join("runs", "segment"))
                            else:
                                # If preserving, rename runs folders with title+resolution+timestamp
                                if bbox_mode:
                                    source_folder = os.path.join("runs", "detect")
                                    destination_folder = os.path.join(
                                        "runs", f"{video_title}_{resolution}_{datetime.now()}"
                                    )
                                    helper.rename_folder(source_folder, destination_folder)
                                if seg_mode:
                                    source_folder = os.path.join("runs", "segment")
                                    destination_folder = os.path.join(
                                        "runs", f"{video_title}_{resolution}_{datetime.now()}"
                                    )
                                    helper.rename_folder(source_folder, destination_folder)

                        # Remove temporary trimmed file (only if tracking_mode was active)
                        if config.tracking_mode:
                            try:
                                os.remove(trimmed_video_path)
                            except FileNotFoundError:
                                pass  # Already removed or wasn't created (no-op)

                    # -----------------------------------------------------------------
                    # Per-video cleanup after all segments processed
                    # -----------------------------------------------------------------
                    if config.external_ssd and processed_flag:
                        # Remove the base file on SSD if we processed at least one segment
                        try:
                            os.remove(base_video_path)
                        except FileNotFoundError:
                            pass
                    if ftp_download:
                        # If video came from FTP, remove base file to save storage
                        try:
                            os.remove(base_video_path)
                        except FileNotFoundError:
                            pass
                    if delete_youtube_video:
                        # Remove the original mp4 from output folder if configured
                        try:
                            os.remove(os.path.join(output_path, f"{vid}.mp4"))
                        except FileNotFoundError:
                            pass

            # ---------------------------------------------------------------------
            # Email notification on successful pass (only if anything was processed)
            # ---------------------------------------------------------------------
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

            # ---------------------------------------------------------------------
            # Sleep between cycles (optional), cleanup mod videos, optionally git pull
            # ---------------------------------------------------------------------
            if config.sleep_sec:
                helper.delete_youtube_mod_videos(video_paths)  # Remove *_mod.mp4 leftovers in video_paths
                logger.info(f"Sleeping for {config.sleep_sec} s before attempting to go over mapping again.")
                time.sleep(config.sleep_sec)

            # Optionally pull latest mapping or code updates
            if config.git_pull:
                common.git_pull()

    # =============================================================================
    # Crash handling: email notification with error details then re-raise
    # =============================================================================
    except Exception as e:
        try:
            if config.email_send:
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Fun image left from original script; kept for continuity
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
            # If email sending fails or config not ready, just swallow and re-raise
            pass
        raise
