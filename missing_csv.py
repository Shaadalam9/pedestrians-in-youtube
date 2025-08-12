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
    """Parse string representing a flat list without quotes into a Python list.

    Supports strings like '[a,b,c]', '[abc]', or '[]'. If already a list, returns as is.

    Args:
        s: The input, which may be a string or a list.

    Returns:
        List containing the parsed items.
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
        s: String containing a valid Python literal list, or a list.

    Returns:
        The evaluated list, or [] if not valid.
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
    """Format a flat list as '[a,b,c]' (no quotes).

    Args:
        lst: List of elements to stringify.

    Returns:
        String representation without quotes.
    """
    return '[' + ','.join(str(x) for x in lst) + ']'


def format_nested_list(lst):
    """Format a nested list as '[[a,b],[c]]' (no quotes).

    Args:
        lst: List of lists to stringify.

    Returns:
        String representation without quotes.
    """
    return '[' + ','.join('[' + ','.join(str(x) for x in inner) + ']' for inner in lst) + ']'


def find_files_to_delete(data_folder):
    """Determine which (video_id, start_time) pairs correspond to CSV files to delete.

    Args:
        data_folder: Directory containing CSV files.

    Returns:
        Set of tuples (video_id, start_time) to be deleted.
    """
    files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    to_delete = set()
    for file in files:
        key = file.replace('.csv', '')
        try:
            video_id, start_time, fps = key.rsplit("_", 2)
            to_delete.add((video_id, int(start_time)))
        except Exception as e:
            logger.warning(f"Skipping file {file}: {e}")
    logger.debug("These video_id, start_time pairs will be removed: %s", to_delete)
    return to_delete


def remove_entry(row, to_delete):
    """Remove video/start_time entries from a row if they're in the to_delete set.

    Args:
        row: DataFrame row.
        to_delete: Set of (video_id, start_time) tuples to remove.

    Returns:
        The updated row with unwanted entries removed.
    """
    # Parse fields from row (strings/lists)
    videos = parse_flat_string_list(row['videos'])
    time_of_day = parse_list(row['time_of_day'])
    start_times = parse_list(row['start_time'])
    end_times = parse_list(row['end_time'])
    vehicle_type = parse_flat_string_list(row['vehicle_type'])
    upload_date = parse_flat_string_list(row['upload_date'])
    channel = parse_flat_string_list(row['channel'])

    # Check consistency for multi-item fields
    if not (len(videos) == len(start_times) == len(end_times) == len(time_of_day)):
        logger.debug("Warning: mismatch in multi fields in row %s", row.get('id', 'unknown'))
        logger.debug("videos: %s", videos)
        logger.debug("start_times: %s", start_times)
        logger.debug("end_times: %s", end_times)
        logger.debug("time_of_day: %s", time_of_day)
        return row
    if not (len(videos) == len(vehicle_type) == len(upload_date) == len(channel)):
        logger.debug("Warning: mismatch in single fields in row %s", row.get('id', 'unknown'))
        logger.debug("videos: %s", videos)
        logger.debug("vehicle_type: %s", vehicle_type)
        logger.debug("upload_date: %s", upload_date)
        logger.debug("channel: %s", channel)
        return row

    # Build new lists for each field, skipping videos/start_times to be deleted
    keep_video_indices = []
    new_time_of_day, new_start_times, new_end_times = [], [], []
    new_vehicle_type, new_upload_date, new_channel = [], [], []

    for i, vid in enumerate(videos):
        these_start_times = start_times[i]
        these_time_of_day = time_of_day[i]
        these_end_times = end_times[i]
        kept_starts, kept_timeofday, kept_end = [], [], []

        for j, st in enumerate(these_start_times):
            if (vid, int(st)) not in to_delete:
                kept_starts.append(these_start_times[j])
                kept_timeofday.append(these_time_of_day[j])
                kept_end.append(these_end_times[j])

        if kept_starts:
            keep_video_indices.append(i)
            new_time_of_day.append(kept_timeofday)
            new_start_times.append(kept_starts)
            new_end_times.append(kept_end)

    # Filter the single-list fields to match remaining videos
    for i in keep_video_indices:
        new_vehicle_type.append(vehicle_type[i])
        new_upload_date.append(upload_date[i])
        new_channel.append(channel[i])

    # Filter videos as well
    new_videos = [videos[i] for i in keep_video_indices]

    # Update row with reformatted lists (as strings for CSV compatibility)
    row['videos'] = format_flat_list(new_videos)
    row['time_of_day'] = format_nested_list(new_time_of_day)
    row['start_time'] = format_nested_list(new_start_times)
    row['end_time'] = format_nested_list(new_end_times)
    row['vehicle_type'] = format_flat_list(new_vehicle_type)
    row['upload_date'] = format_flat_list(new_upload_date)
    row['channel'] = format_flat_list(new_channel)
    return row


def has_any_video(row):
    """Check if the row has any videos left after filtering.

    Args:
        row: DataFrame row.

    Returns:
        True if videos list is non-empty, else False.
    """
    videos = parse_flat_string_list(row['videos'])
    return len(videos) > 0


def main():
    """Main pipeline: Remove specified video/start_times from mapping.csv and save result."""
    # Find which (video_id, start_time) pairs to remove based on CSV files present
    to_delete = find_files_to_delete(DATA_FOLDER)

    # Load mapping.csv as DataFrame
    df = pd.read_csv(MAPPING_PATH)

    # Apply row-wise filtering to remove deleted video/start_times
    df = df.apply(lambda row: remove_entry(row, to_delete), axis=1)

    # Remove rows with no videos remaining
    df = df[df.apply(has_any_video, axis=1)]

    # Save filtered mapping
    filtered_path = "mapping_missing.csv"
    df.to_csv(filtered_path, index=False)
    logger.info(f"Filtered mapping saved to {filtered_path}")


if __name__ == "__main__":
    main()
