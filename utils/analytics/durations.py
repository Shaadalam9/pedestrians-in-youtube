import common
import os
import re
import ast
import heapq
from collections import defaultdict

import polars as pl

from helper_script import Youtube_Helper
from custom_logger import CustomLogger
from utils.core.metadata import MetaData

metadata_class = MetaData()
helper = Youtube_Helper()
logger = CustomLogger(__name__)  # use custom logger

video_paths = common.get_configs("videos")


class Duration:
    def __init__(self) -> None:
        pass

    def get_duration_segment(self, var_dict, df_mapping, name=None,
                             num=common.get_configs('min_max_videos'), duration=None):
        """
        Extract and save video segments based on the fastest tracked objects in provided data.

        This function processes the top-N maximum speed segments (as returned by `find_min_max_video`),
        locates the corresponding video files, computes the actual video segment durations,
        and saves both the trimmed segment and an annotated version with bounding boxes.

        Args:
            var_dict (dict): Nested dictionary of speed values by city/state/condition, video ID, and unique object ID.
            dfs (dict): Dictionary mapping video_start_time keys to DataFrames with tracking data.
            df_mapping (pd.DataFrame): A mapping DataFrame that includes metadata like FPS per video ID.
            num (int): Number of top-speed entries to process.
            duration (float, optional): If provided, use this fixed duration instead of computing from frame count.

        Returns:
            None. Video clips are saved to disk in 'saved_snaps/original' and 'saved_snaps/tracked'.
        """
        data = self.find_min_max_video(var_dict, num=num)
        if num == 0:
            return data

        for segment_type in ["max", "min"]:
            if segment_type in data:
                for city_data in data[segment_type].values():
                    for video_start_time, inner_value in city_data.items():
                        video_name, start_offset = video_start_time.rsplit("_", 1)
                        start_offset = int(start_offset)

                        for unique_id, _ in inner_value.items():
                            try:
                                existing_folder = next(
                                    (
                                        path
                                        for path in video_paths
                                        if os.path.exists(os.path.join(path, f"{video_name}.mp4"))
                                    ),
                                    None,
                                )

                                if not existing_folder:
                                    raise FileNotFoundError(
                                        f"Video file '{video_name}.mp4' not found in any of the specified paths."
                                    )

                                base_video_path = os.path.join(existing_folder, f"{video_name}.mp4")

                                df: pl.DataFrame | None = None

                                for folder_path in common.get_configs("data"):
                                    for subfolder in common.get_configs("sub_domain"):
                                        subfolder_path = os.path.join(folder_path, subfolder)
                                        if not os.path.exists(subfolder_path):
                                            continue
                                        for file in os.listdir(subfolder_path):
                                            if video_start_time in file and file.endswith(".csv"):
                                                file_path = os.path.join(subfolder_path, file)
                                                df = pl.read_csv(file_path)

                                                # Keep only rows with confidence >= min_conf
                                                df = df.filter(
                                                    pl.col("confidence").cast(pl.Float64, strict=False)
                                                    >= float(common.get_configs("min_confidence"))
                                                )
                                                break
                                        if df is not None:
                                            break
                                    if df is not None:
                                        break

                                if df is None:
                                    return None, None

                                filtered_df = df.filter(pl.col("unique-id") == unique_id)

                                if filtered_df.height == 0:
                                    return None, None

                                first_frame = filtered_df.select(pl.col("frame-count").min()).item()
                                last_frame = filtered_df.select(pl.col("frame-count").max()).item()

                                # Lookup fps using mapping (MetaData now Polars-compatible)
                                result = metadata_class.find_values_with_video_id(df_mapping, video_start_time)
                                if result is not None:
                                    fps = result[17]

                                    first_time = first_frame / fps
                                    last_time = last_frame / fps

                                    real_start_time = first_time + start_offset
                                    if duration is None:
                                        real_end_time = start_offset + last_time
                                    else:
                                        real_end_time = real_start_time + duration

                                    helper.trim_video(
                                        input_path=base_video_path,
                                        output_path=os.path.join(
                                            "saved_snaps",
                                            str(name),
                                            segment_type,
                                            "original",
                                            f"{video_name}_{real_start_time}.mp4",
                                        ),
                                        start_time=real_start_time,
                                        end_time=real_end_time,
                                    )

                                    helper.draw_yolo_boxes_on_video(
                                        df=filtered_df,
                                        fps=fps,
                                        video_path=os.path.join(
                                            "saved_snaps",
                                            str(name),
                                            segment_type,
                                            "original",
                                            f"{video_name}_{real_start_time}.mp4",
                                        ),
                                        output_path=os.path.join(
                                            "saved_snaps",
                                            str(name),
                                            segment_type,
                                            "tracked",
                                            f"{video_name}_{real_start_time}.mp4",
                                        ),
                                    )

                            except FileNotFoundError as e:
                                logger.error(f"Error: {e}")

        return data

    def get_duration(self, df, video_id, start_time):
        """
        Retrieve the duration of a specific video segment from a DataFrame.

        Each row in the DataFrame contains information about multiple video segments stored as strings.
        The function looks for a specific `video_id` and `start_time`, and calculates the duration
        by finding the corresponding end time.

        Args:
            df (pd.DataFrame): A DataFrame where each row contains:
                               - 'videos': a string representation of a list of video IDs.
                               - 'start_time': a stringified list of lists of start times.
                               - 'end_time': a stringified list of lists of end times.
            video_id (str): The ID of the video segment to search for.
            start_time (float or int): The specific start time of the video segment.

        Returns:
            float: Duration of the segment (end_time - start_time), or
            None: If no matching segment is found.
        """
        try:
            st_target = int(start_time)
        except Exception:
            return None

        for row in df.select(["videos", "start_time", "end_time"]).iter_rows(named=True):
            try:
                videos_raw = row.get("videos")
                start_raw = row.get("start_time")
                end_raw = row.get("end_time")

                if not isinstance(videos_raw, str) or not isinstance(start_raw, str) or not isinstance(end_raw, str):
                    continue

                videos = re.findall(r"[\w-]+", videos_raw)
                start_times = ast.literal_eval(start_raw)
                end_times = ast.literal_eval(end_raw)

                if video_id not in videos:
                    continue

                idx = videos.index(video_id)
                if not isinstance(start_times, list) or not isinstance(end_times, list):
                    continue
                if idx >= len(start_times) or idx >= len(end_times):
                    continue

                starts_for_vid = start_times[idx]
                ends_for_vid = end_times[idx]
                if not isinstance(starts_for_vid, list) or not isinstance(ends_for_vid, list):
                    continue

                # match start time within that sublist
                if st_target in starts_for_vid:
                    idx_start = starts_for_vid.index(st_target)
                    if idx_start < len(ends_for_vid):
                        end_t = ends_for_vid[idx_start]
                        try:
                            return end_t - st_target
                        except Exception:
                            try:
                                return float(end_t) - float(st_target)
                            except Exception:
                                return None
            except Exception:
                continue

        return None

    def find_min_max_video(self, var_dict, num=2):
        """
        Find the top and bottom N videos based on speed values from a nested dictionary structure.

        The function flattens a deeply nested dictionary containing speed values associated with
        unique identifiers for videos in various city/state/condition groupings. It then extracts
        the `num` largest and smallest speed values and reconstructs a dictionary preserving the original structure.

        Args:
            var_dict (dict): A nested dictionary in the format:
                            {
                                'City_State_Condition': {
                                    'video_id': {
                                        'unique_id': speed_value
                                    }
                                }
                            }
            num (int): Number of top and bottom entries to return based on speed. Default is 2.

        Returns:
            dict: A dictionary with two keys:
                - 'max': contains the top `num` speed entries
                - 'min': contains the bottom `num` speed entries
                Each follows the original nested structure.
        """
        all_speeds = []
        for city_lat_long_cond, videos in var_dict.items():
            for video_id, unique_dict in videos.items():
                for unique_id, speed in unique_dict.items():
                    all_speeds.append((speed, city_lat_long_cond, video_id, unique_id))

        top_n = heapq.nlargest(num, all_speeds, key=lambda x: x[0])
        bottom_n = heapq.nsmallest(num, all_speeds, key=lambda x: x[0])

        def format_result(entries):
            temp_result = defaultdict(lambda: defaultdict(dict))
            for speed, city_lat_long_cond, video_id, unique_id in entries:
                temp_result[city_lat_long_cond][video_id][unique_id] = speed
            return {
                city: {video: dict(uniq) for video, uniq in videos.items()}
                for city, videos in temp_result.items()
            }

        return {"max": format_result(top_n), "min": format_result(bottom_n)}
