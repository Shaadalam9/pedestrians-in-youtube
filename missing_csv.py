import os
import pandas as pd
import ast
import glob
from custom_logger import CustomLogger
from logmod import logs
import common
from analysis import Analysis

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger
analysis_class = Analysis()

# ================================
# Get your folder(s) and mapping file:
FOLDER_PATHS = common.get_configs("data")  # can be a str or list of str
if isinstance(FOLDER_PATHS, str):
    FOLDER_PATHS = [FOLDER_PATHS]
MAPPING_CSV = common.get_configs("mapping")
OUTPUT_CSV = "missing_csv.csv"
# ================================


def delete_video_time_by_filename(df, filename):
    """Removes a specific time entry from all relevant columns, ensuring column sync."""
    filename_no_ext = os.path.splitext(filename)[0]
    video_id, target_time = filename_no_ext.rsplit('_', 1)
    try:
        target_time = int(target_time)
    except ValueError:
        return df

    rows_to_drop = []

    for idx, row in df.iterrows():
        videos = analysis_class.parse_videos(row['videos'])
        times_of_day = ast.literal_eval(row['time_of_day'])
        start_times = ast.literal_eval(row['start_time'])
        end_times = ast.literal_eval(row['end_time'])
        vehicle_type = analysis_class.parse_col(row, 'vehicle_type')
        upload_date = analysis_class.parse_col(row, 'upload_date')
        fps_list = analysis_class.parse_col(row, 'fps_list')
        channel = analysis_class.parse_col(row, 'channel')

        new_videos = []
        new_times_of_day = []
        new_start_times = []
        new_end_times = []
        new_vehicle_type = []
        new_upload_date = []
        new_fps_list = []
        new_channel = []

        for i, vid in enumerate(videos):
            tod = times_of_day[i] if i < len(times_of_day) else []
            sts = start_times[i] if i < len(start_times) else []
            ets = end_times[i] if i < len(end_times) else []

            if vid == video_id:
                keep_indices = [j for j, st in enumerate(sts) if st != target_time]
                new_sts = [sts[j] for j in keep_indices]
                new_tod = [tod[j] for j in keep_indices] if isinstance(tod, list) and len(tod) == len(sts) else tod
                new_ets = [ets[j] for j in keep_indices] if isinstance(ets, list) and len(ets) == len(sts) else ets
                if new_sts:
                    new_videos.append(vid)
                    new_times_of_day.append(new_tod)
                    new_start_times.append(new_sts)
                    new_end_times.append(new_ets)
                    if vehicle_type:
                        new_vehicle_type.append(vehicle_type[i])
                    if upload_date:
                        new_upload_date.append(upload_date[i])
                    if fps_list:
                        new_fps_list.append(fps_list[i])
                    if channel:
                        new_channel.append(channel[i])
                # If not, video and its data are dropped entirely
            else:
                new_videos.append(vid)
                new_times_of_day.append(tod)
                new_start_times.append(sts)
                new_end_times.append(ets)
                if vehicle_type:
                    new_vehicle_type.append(vehicle_type[i])
                if upload_date:
                    new_upload_date.append(upload_date[i])
                if fps_list:
                    new_fps_list.append(fps_list[i])
                if channel:
                    new_channel.append(channel[i])

        # Remove row if no videos left
        if not new_videos:
            rows_to_drop.append(idx)
            continue

        # Always keep columns in sync
        df.at[idx, 'videos'] = "[" + ",".join(new_videos) + "]"
        df.at[idx, 'time_of_day'] = str(new_times_of_day)
        df.at[idx, 'start_time'] = str(new_start_times)
        df.at[idx, 'end_time'] = str(new_end_times)
        if vehicle_type:
            df.at[idx, 'vehicle_type'] = str(new_vehicle_type)
        if upload_date:
            df.at[idx, 'upload_date'] = str(new_upload_date)
        if fps_list:
            df.at[idx, 'fps_list'] = str(new_fps_list)
        if channel:
            df.at[idx, 'channel'] = str(new_channel)

    df = df.drop(index=rows_to_drop).reset_index(drop=True)
    return df


def process_folder(folder_paths, mapping_csv, output_csv):
    """Scan all folders, check for video csvs with Frame Count column, and update mapping."""
    df = pd.read_csv(mapping_csv)

    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    present_files = set()
    for folder_path in folder_paths:
        logger.debug(f"Scanning folder: {folder_path}")
        for file in os.listdir(folder_path):
            if ".csv" not in file:
                continue
            base_name = analysis_class.clean_csv_filename(file)
            present_files.add(base_name)

    logger.debug(f"Files found (unique .csv base): {len(present_files)}")
    for csv_file in present_files:
        file_base = os.path.splitext(csv_file)[0]
        if "_" not in file_base:
            continue
        try:
            _ = int(file_base.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue

        found = False
        for folder_path in folder_paths:
            pattern = os.path.join(folder_path, file_base + ".csv*")
            for path in glob.glob(pattern):
                found = True
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        header_line = f.readline()
                    header_columns = [col.strip().strip('"') for col in header_line.strip().split(",")]
                except Exception as e:
                    logger.error(f"Could not read header for {path}: {e}")
                    continue
                if "Frame Count" in header_columns:
                    logger.info(f"Removing info for: {csv_file} (Frame Count found)")
                    df = delete_video_time_by_filename(df, csv_file)
                else:
                    logger.info(f"Skipping {csv_file} ({os.path.basename(path)}): No Frame Count")
                break  # Only process the first matching file
            if found:
                break
        if not found:
            logger.info(f"Could not find file for {csv_file} in any folder.")

    df.to_csv(output_csv, index=False)
    logger.info(f"Saved cleaned mapping to: {output_csv}")


if __name__ == "__main__":
    process_folder(FOLDER_PATHS, MAPPING_CSV, OUTPUT_CSV)
