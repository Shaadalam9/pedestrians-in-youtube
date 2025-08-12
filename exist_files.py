import os
import pandas as pd
import ast
import common
from custom_logger import CustomLogger
from logmod import logs

# Initialize logging according to config
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

# Directory containing per-frame CSV data files
DATA_FOLDER = common.get_configs("data")
MAPPING_PATH = common.get_configs("mapping")


def parse_flat_string_list(s):
    """Parse a string representing a flat list without quotes into a Python list.

    Supports '[a,b,c]', '[abc]', or '[]'. Returns as-is if already a list.

    Args:
        s: Input string or list.

    Returns:
        list: Parsed Python list.
    """
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [v.strip() for v in inner.split(',')]
    if s == '':
        return []
    return [s]


def parse_list(s):
    """Parse a string representing a (possibly nested) Python list.

    Args:
        s: Input string or list.

    Returns:
        list: Python-evaluated list, or [] if parsing fails.
    """
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    s = s.strip()
    if s == "":
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def format_flat_list(lst):
    """Format a flat list as a string like '[a,b,c]' (no quotes).

    Args:
        lst: List to stringify.

    Returns:
        str: String representation.
    """
    return '[' + ','.join(str(x) for x in lst) + ']'


def format_nested_list(lst):
    """Format a nested list as a string like '[[a,b],[c]]' (no quotes).

    Args:
        lst: List of lists.

    Returns:
        str: String representation.
    """
    return '[' + ','.join('[' + ','.join(str(x) for x in inner) + ']' for inner in lst) + ']'


def gather_present_pairs(data_folder):
    """Gather all present (video_id, start_time) pairs from CSV filenames in the folder.

    Args:
        data_folder: Directory containing .csv files.

    Returns:
        set: Set of (video_id, start_time) tuples.
    """
    files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    present = set()
    for file in files:
        key = file.replace('.csv', '')
        try:
            video_id, start_time, fps = key.rsplit("_", 2)
            present.add((video_id, int(start_time)))
        except Exception as e:
            logger.info(f"Skipping file {file}: {e}")
    logger.debug("Only these video_id, start_time pairs will be KEPT:", present)
    return present


def keep_present_entry(row, present):
    """Filter out entries in a mapping row that are not in the 'present' set.

    Args:
        row: DataFrame row.
        present: Set of (video_id, start_time) pairs to keep.

    Returns:
        The updated row with only present entries.
    """
    # Parse fields from row (strings/lists)
    videos = parse_flat_string_list(row['videos'])
    time_of_day = parse_list(row['time_of_day'])
    start_times = parse_list(row['start_time'])
    end_times = parse_list(row['end_time'])
    vehicle_type = parse_flat_string_list(row['vehicle_type'])
    upload_date = parse_flat_string_list(row['upload_date'])
    channel = parse_flat_string_list(row['channel'])

    # Sanity checks
    if not (len(videos) == len(start_times) == len(end_times) == len(time_of_day)):
        logger.error(f"Warning: mismatch in multi fields in row {row.get('id', 'unknown')}")
        logger.error(f"videos: {videos}")
        logger.error(f"start_times: {start_times}")
        logger.error(f"end_times: {end_times}")
        logger.error(f"time_of_day: {time_of_day}")
        return row
    if not (len(videos) == len(vehicle_type) == len(upload_date) == len(channel)):
        logger.error(f"Warning: mismatch in single fields in row {row.get('id', 'unknown')}")
        logger.error(f"videos: {videos}")
        logger.error(f"vehicle_type: {vehicle_type}")
        logger.error(f"upload_date: {upload_date}")
        logger.error(f"channel: {channel}")
        return row

    keep_video_indices = []
    new_time_of_day, new_start_times, new_end_times = [], [], []
    new_vehicle_type, new_upload_date, new_channel = [], [], []

    for i, vid in enumerate(videos):
        these_start_times = start_times[i]
        these_time_of_day = time_of_day[i]
        these_end_times = end_times[i]
        kept_starts, kept_timeofday, kept_end = [], [], []

        for j, st in enumerate(these_start_times):
            if (vid, int(st)) in present:
                kept_starts.append(these_start_times[j])
                kept_timeofday.append(these_time_of_day[j])
                kept_end.append(these_end_times[j])

        if kept_starts:
            keep_video_indices.append(i)
            new_time_of_day.append(kept_timeofday)
            new_start_times.append(kept_starts)
            new_end_times.append(kept_end)

    # Filter single-list fields
    for i in keep_video_indices:
        new_vehicle_type.append(vehicle_type[i])
        new_upload_date.append(upload_date[i])
        new_channel.append(channel[i])

    new_videos = [videos[i] for i in keep_video_indices]

    # Update row with filtered lists
    row['videos'] = format_flat_list(new_videos)
    row['time_of_day'] = format_nested_list(new_time_of_day)
    row['start_time'] = format_nested_list(new_start_times)
    row['end_time'] = format_nested_list(new_end_times)
    row['vehicle_type'] = format_flat_list(new_vehicle_type)
    row['upload_date'] = format_flat_list(new_upload_date)
    row['channel'] = format_flat_list(new_channel)
    return row


def has_any_video(row):
    """Check if the row has any videos after filtering.

    Args:
        row: DataFrame row.

    Returns:
        bool: True if any videos remain, else False.
    """
    videos = parse_flat_string_list(row['videos'])
    return len(videos) > 0


def main():
    """Main function: Filters mapping.csv to keep only present (video_id, start_time) pairs."""
    present = gather_present_pairs(DATA_FOLDER)
    df = pd.read_csv(MAPPING_PATH)

    # Filter entries in DataFrame
    df = df.apply(lambda row: keep_present_entry(row, present), axis=1)
    # Remove rows with no videos left
    df = df[df.apply(has_any_video, axis=1)]

    filtered_path = "mapping_exist.csv"
    df.to_csv(filtered_path, index=False)
    logger.info(f"Filtered mapping saved to {filtered_path}")


if __name__ == "__main__":
    main()
