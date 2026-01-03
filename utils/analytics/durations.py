import common
import os
import re
import ast
from collections import defaultdict
import heapq
from helper_script import Youtube_Helper
import pandas as pd
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

        # Process only the 'max' speed segments
        for segment_type in ['max', 'min']:
            if segment_type in data:
                for city_data in data[segment_type].values():
                    for video_start_time, inner_value in city_data.items():
                        # Extract base video name and its offset
                        video_name, start_offset = video_start_time.rsplit('_', 1)
                        start_offset = int(start_offset)

                        for unique_id, _ in inner_value.items():
                            try:
                                # Find the existing folder containing the video file
                                existing_folder = next((
                                    path for path in video_paths if os.path.exists(
                                        os.path.join(path, f"{video_name}.mp4"))), None)

                                if not existing_folder:
                                    raise FileNotFoundError(f"Video file '{video_name}.mp4' not found in any of the specified paths.")  # noqa:E501

                                base_video_path = os.path.join(existing_folder, f"{video_name}.mp4")

                                df = None  # Initialize before the loops

                                for folder_path in common.get_configs('data'):
                                    for subfolder in common.get_configs("sub_domain"):
                                        subfolder_path = os.path.join(folder_path, subfolder)
                                        if not os.path.exists(subfolder_path):
                                            continue
                                        for file in os.listdir(subfolder_path):
                                            if video_start_time in file and file.endswith('.csv'):
                                                file_path = os.path.join(subfolder_path, file)
                                                df = pd.read_csv(file_path)

                                                # Keep only rows with confidence > min_conf
                                                df = df[df["confidence"] >= common.get_configs("min_confidence")]
                                                break  # Found the file, break from subfolder loop
                                        if df is not None:
                                            break  # Break from folder_path loop if found
                                    if df is not None:
                                        break

                                if df is None:
                                    return None, None  # Could not find any matching CSV

                                filtered_df = df[df['unique-id'] == unique_id]

                                if filtered_df.empty:
                                    return None, None  # No data found for this unique_id

                                # Determine frame-based start and end times
                                first_frame = filtered_df['frame-count'].min()
                                last_frame = filtered_df['frame-count'].max()

                                # Look up the frame rate (fps) using the video_start_time
                                result = metadata_class.find_values_with_video_id(df_mapping, video_start_time)

                                # Check if the result is None (i.e., no matching data was found)
                                if result is not None:
                                    # Unpack the result since it's not None
                                    fps = result[17]

                                    first_time = first_frame / fps
                                    last_time = last_frame / fps

                                    # Adjusted start and end times
                                    real_start_time = first_time + start_offset
                                    if duration is None:
                                        real_end_time = start_offset + last_time
                                    else:
                                        real_end_time = real_start_time + duration

                                    # Trim and save the raw segment
                                    helper.trim_video(
                                        input_path=base_video_path,
                                        output_path=os.path.join("saved_snaps",
                                                                 str(name),
                                                                 segment_type,
                                                                 "original",
                                                                 f"{video_name}_{real_start_time}.mp4"),
                                        start_time=real_start_time,
                                        end_time=real_end_time
                                    )

                                    # Overlay YOLO boxes on the saved segment
                                    helper.draw_yolo_boxes_on_video(df=filtered_df,
                                                                    fps=fps,
                                                                    video_path=os.path.join("saved_snaps",
                                                                                            str(name),
                                                                                            segment_type,
                                                                                            "original",
                                                                                            f"{video_name}_{real_start_time}.mp4"),  # noqa:E501
                                                                    output_path=os.path.join("saved_snaps",
                                                                                             str(name),
                                                                                             segment_type,
                                                                                             "tracked",
                                                                                             f"{video_name}_{real_start_time}.mp4"))  # noqa:E501

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
        for _, row in df.iterrows():
            # Extract list of video IDs
            videos = re.findall(r"[\w-]+", row["videos"])

            # Convert stringified lists of lists into Python objects
            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])

            if video_id in videos:
                index = videos.index(video_id)  # get the index of the video

                # Ensure start_time is in the corresponding start_times list
                if start_time in start_times[index]:  # check if start_time matches
                    # find end time that matches the start time
                    index_start = start_times[index].index(start_time)
                    end_time = end_times[index][index_start]
                    # Return the difference as the duration
                    return end_time - start_time

        # Return None if no matching video_id and start_time were found
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

        # Flatten the nested dictionary and collect tuples of (speed, city_state_cond, video_id, unique_id)
        for city_lat_long_cond, videos in var_dict.items():
            for video_id, unique_dict in videos.items():
                for unique_id, speed in unique_dict.items():
                    all_speeds.append((speed, city_lat_long_cond, video_id, unique_id))

        # Use heapq to efficiently get the top and bottom N entries based on speed
        top_n = heapq.nlargest(num, all_speeds, key=lambda x: x[0])
        bottom_n = heapq.nsmallest(num, all_speeds, key=lambda x: x[0])

        # Helper function to rebuild the nested structure from a list of tuples
        def format_result(entries):
            temp_result = defaultdict(lambda: defaultdict(dict))
            for speed, city_lat_long_cond, video_id, unique_id in entries:
                temp_result[city_lat_long_cond][video_id][unique_id] = speed

            # Convert defaultdicts to regular dicts for clean output
            return {
                city: {video: dict(uniq) for video, uniq in videos.items()}
                for city, videos in temp_result.items()
            }

        # Return both top and bottom results in the original structure
        return {
            'max': format_result(top_n),
            'min': format_result(bottom_n)
        }
