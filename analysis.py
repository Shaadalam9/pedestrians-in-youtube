# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import math
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import heapq
from helper_script import Youtube_Helper
from utils.algorithms import Algorithms
from utils.values import Values
from utils.wrappers import Wrappers
from utils.plot import Plots
from utils.geometry import Geometry
from utils.tools import Tools
import common
from custom_logger import CustomLogger
from logmod import logs
import ast
import pickle
from tqdm import tqdm
import re
import warnings
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from datetime import datetime
from typing import ClassVar, Dict, Any, Optional, Set


# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

helper = Youtube_Helper()
algorithms_class = Algorithms()
values_class = Values()
wrapper_class = Wrappers()
plots_class = Plots()
geometry_class = Geometry()
tools_class = Tools()

# set template for plotly output
template = common.get_configs('plotly_template')

# File to store the city coordinates
file_results = 'results.pickle'

# Colours in graphs
bar_colour_1 = 'rgb(251, 180, 174)'
bar_colour_2 = 'rgb(179, 205, 227)'
bar_colour_3 = 'rgb(204, 235, 197)'
bar_colour_4 = 'rgb(222, 203, 228)'

# Consts
BASE_HEIGHT_PER_ROW = 20  # Adjust as needed
SCALE = 1  # scale=3 hangs often

video_paths = common.get_configs("videos")
misc_files: set[str] = {"DS_Store", "seg", "bbox"}  # define once


class Analysis():

    def __init__(self) -> None:
        pass

    def filter_csv_files(self, file, df_mapping):
        """
        Filters and processes CSV files based on predefined criteria.

        This function checks if the given file is a CSV, verifies its mapping and value requirements,
        and further processes the file by loading it into a DataFrame and optionally applying geometry corrections.
        Files are only accepted if their mapping indicates sufficient footage and if required columns are present.

        Args:
            file (str): The filename to check and process.

        Returns:
            str or None: The original filename if all checks pass and the file is valid for processing;
                         otherwise, None to indicate the file should be skipped.

        Notes:
            - This method depends on several external classes and variables:
                - `values_class`: For value lookup and calculations.
                - `df_mapping`: DataFrame with mapping data for video IDs.
                - `common`: Configuration utility for various thresholds and flags.
                - `geometry_class`: Utility for geometry correction.
                - `logger`: Logging utility.
                - `folder_path`: Path to search for CSV files.
        """
        # Only process files ending with ".csv"
        file = tools_class.clean_csv_filename(file)
        if file.endswith(".csv"):
            filename = os.path.splitext(file)[0]

            # Lookup values *before* reading CSV
            values = values_class.find_values_with_video_id(df_mapping, filename)
            if values is None:
                return None  # Skip if mapping or required value is None

            vehicle_type = values[18]
            vehicle_list = common.get_configs("vehicles_analyse")

            # Only check if the list is NOT empty
            if vehicle_list:  # This is True if the list is not empty
                if vehicle_type not in vehicle_list:
                    return None

            # # Check if the footage duration meets the minimum threshold
            # total_seconds = values_class.calculate_total_seconds_for_city(
            #     df_mapping, values[4], values[5]
            # )

            # if total_seconds <= common.get_configs("footage_threshold"):
            #     return None  # Skip if not enough seconds

            # file_path = os.path.join(folder_path, file)
            # try:
            #     logger.debug(f"Adding file {file_path} to dfs.")

            #     # Read the CSV into a DataFrame
            #     df = pd.read_csv(file_path)

            #     # Optionally apply geometry correction if configured and not zero
            #     use_geom_correction = common.get_configs("use_geometry_correction")
            #     if use_geom_correction != 0:
            #         df = geometry_class.reassign_ids_directional_cross_fix(
            #             df,
            #             distance_threshold=use_geom_correction,
            #             yolo_ids=[0]
            #         )
            # except Exception as e:
            #     logger.error(f"Failed to read {file_path}: {e}.")
            #     return  # Skip to the next file if reading fails
        return file

    def parse_videos(self, s):
        """Parse a bracketed, comma-separated video string into a list of IDs.

        Args:
            s (str): String representing video IDs, e.g. '[abc,def,ghi]'.

        Returns:
            List[str]: List of video IDs as strings, e.g. ['abc', 'def', 'ghi'].

        Example:
            >>> self.parse_videos('[abc,def]')
            ['abc', 'def']
        """
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return [x.strip() for x in s.split(",") if x.strip()]

    def parse_col(self, row, colname):
        """Safely parse a DataFrame row column (stored as a string) to a Python object.

        Args:
            row (pd.Series): The DataFrame row containing the column.
            colname (str): The column name to parse.

        Returns:
            object: The parsed Python object (e.g., list or int). Returns empty list on failure.

        Example:
            >>> self.parse_col(row, 'start_time')
            [[12], [34], [56]]
        """
        try:
            return ast.literal_eval(row[colname])
        except (ValueError, SyntaxError, KeyError, TypeError):
            return []

    def delete_video_time_by_filename(self, df, filename, output_file=None):
        """Remove a specific time entry from a video's lists in the mapping DataFrame.

        For each row, finds the matching video and start_time (from the provided filename),
        removes the corresponding time-of-day, start_time, and end_time entry. If a video
        ends up with no times, it is fully removed from all per-video columns. If a row
        is left with no videos, the row is dropped from the DataFrame.

        Args:
            df (pd.DataFrame): The mapping DataFrame to update.
            filename (str): Name of the CSV/video file, e.g. 'l72z2l_1h9A_7645.csv',
                used to extract the video ID and start_time to remove. The video ID may
                contain underscores.
            output_file (str, optional): If provided, writes the cleaned DataFrame to this path.

        Returns:
            pd.DataFrame: The updated DataFrame with the specified entry removed.

        Example:
            >>> df = self.delete_video_time_by_filename(df, 'l72z2l_1h9A_7645.csv')
        """

        # Clean filename if necessary and extract video_id and target_time (last underscore split)
        filename = tools_class.clean_csv_filename(filename)
        filename_no_ext = os.path.splitext(filename)[0]
        video_id, target_time = filename_no_ext.rsplit('_', 1)
        target_time = int(target_time)

        rows_to_drop = []

        for idx, row in df.iterrows():
            # Parse video list as in original format: [id1,id2,...] with no quotes
            videos = self.parse_videos(row['videos'])
            times_of_day = ast.literal_eval(row['time_of_day'])
            start_times = ast.literal_eval(row['start_time'])
            end_times = ast.literal_eval(row['end_time'])

            changed = False  # Track if this row has been modified

            # Prepare new lists for updating the row
            new_videos = []
            new_times_of_day = []
            new_start_times = []
            new_end_times = []
            new_vehicle_type = []
            new_upload_date = []
            new_fps_list = []
            new_channel = []

            # Parse per-video columns (if present, else empty lists)
            vehicle_type = self.parse_col(row, 'vehicle_type')
            upload_date = self.parse_col(row, 'upload_date')
            fps_list = self.parse_col(row, 'fps_list')
            channel = self.parse_col(row, 'channel')

            # Loop through each video for this row
            for i, vid in enumerate(videos):
                # Get per-time lists for this video (safely)
                tod = times_of_day[i] if i < len(times_of_day) else []
                sts = start_times[i] if i < len(start_times) else []
                ets = end_times[i] if i < len(end_times) else []

                if vid == video_id:
                    # Find indices where the start_time does NOT match the target
                    keep_indices = [j for j, st in enumerate(sts) if st != target_time]
                    # Keep only the unmatched entries in all per-time lists
                    new_sts = [sts[j] for j in keep_indices]
                    new_tod = [tod[j] for j in keep_indices] if isinstance(tod, list) and len(tod) == len(sts) else tod
                    new_ets = [ets[j] for j in keep_indices] if isinstance(ets, list) and len(ets) == len(sts) else ets

                    if new_sts:
                        # Still times left for this video, keep it and related info
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
                    else:
                        # No times left, fully remove this video from all columns
                        changed = True
                else:
                    # Unrelated video, keep as-is
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

            # If all videos are gone for this row, mark row for dropping
            if len(new_videos) == 0:
                rows_to_drop.append(idx)
                continue

            # If something changed, update the row
            if changed or len(new_videos) != len(videos):
                # Write videos in original CSV style: [id1,id2,id3]
                df.at[idx, 'videos'] = "[" + ",".join(new_videos) + "]"
                # Write per-time columns in str(list) (keeps double brackets)
                df.at[idx, 'time_of_day'] = str(new_times_of_day)
                df.at[idx, 'start_time'] = str(new_start_times)
                df.at[idx, 'end_time'] = str(new_end_times)

                # Only update per-video columns if present (may be missing in some datasets)
                if vehicle_type:
                    df.at[idx, 'vehicle_type'] = str(new_vehicle_type)
                if upload_date:
                    df.at[idx, 'upload_date'] = str(new_upload_date)
                if fps_list:
                    df.at[idx, 'fps_list'] = str(new_fps_list)
                if channel:
                    df.at[idx, 'channel'] = str(new_channel)

        # Drop rows where all videos were removed
        df = df.drop(index=rows_to_drop).reset_index(drop=True)

        # Save to CSV if requested
        if output_file:
            df.to_csv(output_file, index=False)
        return df

    @staticmethod
    def get_unique_values(df, value):
        """
        Calculates the number of unique countries from a DataFrame.

        Args:
            df (DataFrame): A DataFrame containing the CSV data.

        Returns:
            tuple: A set of unique countries and the total count of unique countries.
        """
        unique_values = set(df[value].unique())

        return unique_values, len(unique_values)

    # class-level cache to store metrics for all video files, avoids redundant computation
    _all_metrics_cache: ClassVar[Dict[str, Any]] = {}

    @classmethod
    def _compute_all_metrics(cls, df_mapping: pd.DataFrame) -> None:
        """
        Computes and caches all traffic and object detection metrics for the given video mapping DataFrame.

        This method processes mapping information for video files, scans associated CSV detection results,
        computes summary metrics per video (such as cell phone usage per person, object counts per minute for
        various vehicle types, traffic signs, and persons), and caches the results grouped and wrapped by city/country.
        The results are stored in a class-level cache to avoid repeated computation.

        Args:
            df_mapping (pandas.DataFrame): A DataFrame containing metadata about video files,
                with columns expected to include 'videos', 'start_time', and 'time_of_day'.
                Each row describes a set of related video segments.

        Side Effects:
            Updates the class-level cache variable `_all_metrics_cache` with a dictionary containing
            all metrics, each mapped via `wrapper_class.city_country_wrapper` for aggregation.

        Metrics computed (keys in cache):
            - "cellphones": cell phone detections per person, normalized for time
            - "traffic_signs": count of detected traffic signs (YOLO ids 9, 11)
            - "vehicles": count of all vehicles (YOLO ids 2, 3, 5, 7)
            - "bicycles": count of bicycles (YOLO id 1)
            - "cars": count of cars (YOLO id 2)
            - "motorcycles": count of motorcycles (YOLO id 3)
            - "buses": count of buses (YOLO id 5)
            - "trucks": count of trucks (YOLO id 7)
            - "persons": count of people (YOLO id 0)
        """

        # List of data folders containing detection CSVs
        data_folders = common.get_configs('data')
        csv_files: Dict[str, str] = {}

        # Index all CSV files from bbox and seg subfolders for quick lookup
        for folder_path in data_folders:
            for subfolder in common.get_configs("sub_domain"):
                subfolder_path = os.path.join(folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    continue
                for file in os.listdir(subfolder_path):
                    if file.endswith('.csv'):
                        csv_files[file] = os.path.join(subfolder_path, file)

        # Prepare result containers for each metric type
        cellphone_info: Dict[str, float] = {}
        traffic_signs_layer: Dict[str, float] = {}
        vehicle_layer: Dict[str, float] = {}
        bicycle_layer: Dict[str, float] = {}
        car_layer: Dict[str, float] = {}
        motorcycle_layer: Dict[str, float] = {}
        bus_layer: Dict[str, float] = {}
        truck_layer: Dict[str, float] = {}
        person_layer: Dict[str, float] = {}

        # Process each mapping row (one or more videos per row)
        for _, row in tqdm(df_mapping.iterrows(), total=df_mapping.shape[0], desc="Analysing the csv files:"):
            video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
            start_times = ast.literal_eval(row["start_time"])
            time_of_day = ast.literal_eval(row["time_of_day"])

            # Loop through all video_id + start_time pairs
            for vid, start_times_list, time_of_day_list in zip(video_ids, start_times, time_of_day):
                for start_time, time_of_day_value in zip(start_times_list, time_of_day_list):
                    prefix = f"{vid}_{start_time}_"
                    # Find the file whose name starts with this prefix
                    matching_files = [fname for fname in csv_files if fname.startswith(prefix) and fname.endswith('.csv')]  # noqa:E501
                    if not matching_files:
                        logger.warning(f"[WARNING] File not found for prefix: {prefix}")
                        continue
                    elif len(matching_files) > 1:
                        logger.warning(f"[WARNING] Multiple files found for prefix: {prefix}, using the first one: {matching_files[0]}")  # noqa:E501
                    filename = matching_files[0]

                    # Extract fps using regex
                    match = re.match(rf"{vid}_{start_time}_(\d+)\.csv", filename)
                    if match:
                        fps = int(match.group(1))
                    else:
                        logger.error(f"[ERROR] Could not extract fps from filename: {filename}")
                        continue

                    filename = f"{vid}_{start_time}_{fps}.csv"
                    if filename not in csv_files:
                        continue  # No detection CSV for this video segment

                    file_path = csv_files[filename]

                    # Find video meta details (start, end, city, location, etc.)
                    result = values_class.find_values_with_video_id(df_mapping, f"{vid}_{start_time}_{fps}")
                    if result is None:
                        continue

                    start = result[1]
                    end = result[2]
                    condition = result[3]
                    city = result[4]
                    lat = result[6]
                    long = result[7]
                    fps = result[17]
                    duration = end - start  # Duration in seconds
                    # city_id_format is not used later, so we keep it for clarity / potential future use
                    city_id_format = f'{city}_{lat}_{long}_{condition}'  # noqa: F841
                    video_key = f"{vid}_{start_time}_{fps}"

                    # Load detection data for this video segment
                    dataframe = pd.read_csv(file_path)

                    # ---- CELL PHONES: Count per person, normalised ----
                    mobile_ids = len(dataframe[dataframe["yolo-id"] == 67]["unique-id"].unique())
                    num_person = len(dataframe[dataframe["yolo-id"] == 0]["unique-id"].unique())
                    if num_person > 0 and mobile_ids > 0 and duration > 0:
                        avg_cellphone = ((mobile_ids * 60) / duration / num_person) * 1000
                        cellphone_info[video_key] = float(avg_cellphone)

                    # ---- TRAFFIC SIGNS (YOLO 9, 11) ----
                    traffic_sign_ids = dataframe[dataframe["yolo-id"].isin([9, 11])]["unique-id"].unique()
                    count = (len(traffic_sign_ids) / duration) * 60 if duration > 0 else 0.0
                    traffic_signs_layer[video_key] = float(count)

                    # ---- VEHICLES (YOLO 2,3,5,7) ----
                    vehicles_mask = dataframe["yolo-id"].isin([2, 3, 5, 7])
                    vehicle_ids = dataframe[vehicles_mask]["unique-id"].unique()
                    count = (len(vehicle_ids) / duration) * 60 if duration > 0 else 0.0
                    vehicle_layer[video_key] = float(count)

                    # ---- BICYCLES (YOLO 1) ----
                    bicycle_ids = dataframe[dataframe["yolo-id"] == 1]["unique-id"].unique()
                    count = (len(bicycle_ids) / duration) * 60 if duration > 0 else 0.0
                    bicycle_layer[video_key] = float(count)

                    # ---- CARS (YOLO 2) ----
                    car_ids = dataframe[dataframe["yolo-id"] == 2]["unique-id"].unique()
                    count = (len(car_ids) / duration) * 60 if duration > 0 else 0.0
                    car_layer[video_key] = float(count)

                    # ---- MOTORCYCLES (YOLO 3) ----
                    motorcycle_ids = dataframe[dataframe["yolo-id"] == 3]["unique-id"].unique()
                    count = (len(motorcycle_ids) / duration) * 60 if duration > 0 else 0.0
                    motorcycle_layer[video_key] = float(count)

                    # ---- BUSES (YOLO 5) ----
                    bus_ids = dataframe[dataframe["yolo-id"] == 5]["unique-id"].unique()
                    count = (len(bus_ids) / duration) * 60 if duration > 0 else 0.0
                    bus_layer[video_key] = float(count)

                    # ---- TRUCKS (YOLO 7) ----
                    truck_ids = dataframe[dataframe["yolo-id"] == 7]["unique-id"].unique()
                    count = (len(truck_ids) / duration) * 60 if duration > 0 else 0.0
                    truck_layer[video_key] = float(count)

                    # ---- PERSONS (YOLO 0) ----
                    person_ids = dataframe[dataframe["yolo-id"] == 0]["unique-id"].unique()
                    count = (len(person_ids) / duration) * 60 if duration > 0 else 0.0
                    person_layer[video_key] = float(count)

        # --- WRAPPING AS CITY_LONGITUDE_LATITUDE_CONDITION ---
        metric_dicts = [
            ("cellphones", cellphone_info),
            ("traffic_signs", traffic_signs_layer),
            ("vehicles", vehicle_layer),
            ("bicycles", bicycle_layer),
            ("cars", car_layer),
            ("motorcycles", motorcycle_layer),
            ("buses", bus_layer),
            ("trucks", truck_layer),
            ("persons", person_layer),
        ]

        # Reinitialize cache on recompute
        cls._all_metrics_cache = {}

        for i, (metric_name, metric_layer) in enumerate(metric_dicts, 1):
            logger.info(f"[{i}/{len(metric_dicts)}] Wrapping '{metric_name}' ...")
            wrapped = wrapper_class.city_country_wrapper(
                input_dict=metric_layer,
                mapping=df_mapping,
                show_progress=True
            )
            cls._all_metrics_cache[metric_name] = wrapped

    @classmethod
    def _ensure_cache(cls, df_mapping: pd.DataFrame) -> None:
        """
        Ensure that the class-level metrics cache is populated.
        If the cache is empty, computes all metrics for the provided mapping DataFrame.
        """
        if not cls._all_metrics_cache:
            cls._compute_all_metrics(df_mapping)

    @classmethod
    def calculate_cellphones(cls, df_mapping: pd.DataFrame):
        """
        Return the cached cell phone metric, computing all metrics if needed.
        """
        cls._ensure_cache(df_mapping)
        return cls._all_metrics_cache["cellphones"]

    @classmethod
    def calculate_traffic_signs(cls, df_mapping: pd.DataFrame):
        """
        Return the cached traffic sign metric, computing all metrics if needed.
        """
        cls._ensure_cache(df_mapping)
        return cls._all_metrics_cache["traffic_signs"]

    @classmethod
    def calculate_traffic(
        cls,
        df_mapping: pd.DataFrame,
        person: int = 0,
        bicycle: int = 0,
        motorcycle: int = 0,
        car: int = 0,
        bus: int = 0,
        truck: int = 0,
    ):
        """
        Return the requested vehicle/person/bicycle metric from the cache, computing if needed.
        Arguments specify which traffic metric to return. If multiple flags are set, precedence is given as:
        - 'person' if set
        - 'bicycle' if set
        - if all of motorcycle, car, bus, truck are set: returns 'vehicles'
        - otherwise, returns individual type if its flag is set
        - fallback is 'vehicles'
        """
        cls._ensure_cache(df_mapping)

        if person:
            return cls._all_metrics_cache["persons"]
        if bicycle:
            return cls._all_metrics_cache["bicycles"]
        if motorcycle and car and bus and truck:
            return cls._all_metrics_cache["vehicles"]
        if car:
            return cls._all_metrics_cache["cars"]
        if motorcycle:
            return cls._all_metrics_cache["motorcycles"]
        if bus:
            return cls._all_metrics_cache["buses"]
        if truck:
            return cls._all_metrics_cache["trucks"]
        # Fallback to all vehicles
        return cls._all_metrics_cache["vehicles"]

    # Optional helper to force a rebuild, e.g. after data changes
    @classmethod
    def clear_cache(cls) -> None:
        cls._all_metrics_cache.clear()

    @staticmethod
    def crossing_event_with_traffic_equipment(df_mapping, data):
        """
        Analyse pedestrian crossing events in relation to the presence of traffic equipment (yolo-id 9 or 11).

        For each video and crossing, counts are computed for crossings where
        relevant traffic equipment was present or absent during the crossing.
        Aggregates counts and total video durations by city/condition and country/condition.

        Args:
            df_mapping (dict): Mapping of video keys to relevant metadata.
            data (dict): Dictionary of DataFrames containing pedestrian crossing data for each video.

        Returns:
            tuple: (
                crossings_with_traffic_equipment_city (dict): Counts of crossings with equipment per city/condition,
                crossings_without_traffic_equipment_city (dict):
                        Counts of crossings without equipment per city/condition,
                total_duration_by_city (dict): Total duration (seconds) per city/condition,
                crossings_with_traffic_equipment_country (dict):
                        Counts of crossings with equipment per country/condition,
                crossings_without_traffic_equipment_country (dict):
                        Counts of crossings without equipment per country/condition,
                total_duration_by_country (dict): Total duration (seconds) per country/condition
            )
        """
        total_duration_by_city = {}
        total_duration_by_country = {}
        crossings_with_traffic_equipment_city = {}
        crossings_with_traffic_equipment_country = {}
        crossings_without_traffic_equipment_city = {}
        crossings_without_traffic_equipment_country = {}

        for video_key, crossings in tqdm(data.items(), total=len(data)):
            count_with_equipment = 0
            count_without_equipment = 0

            # Extract metadata for this video
            result = values_class.find_values_with_video_id(df_mapping, video_key)
            if result is not None:
                start_time = result[1]
                end_time = result[2]
                condition = result[3]
                city = result[4]
                latitude = result[6]
                longitude = result[7]
                country = result[8]

                location_key_city = f'{city}_{latitude}_{longitude}_{condition}'
                location_key_country = f'{country}_{condition}'

                # Update total duration per location/condition
                duration = end_time - start_time
                total_duration_by_city[location_key_city] = total_duration_by_city.get(location_key_city, 0) + duration
                total_duration_by_country[location_key_country] = total_duration_by_country.get(
                    location_key_country, 0) + duration

                # Find the CSV file for this video
                value = None
                for folder_path in common.get_configs('data'):
                    for subfolder in common.get_configs("sub_domain"):
                        subfolder_path = os.path.join(folder_path, subfolder)
                        if not os.path.exists(subfolder_path):
                            continue  # Skip if subfolder doesn't exist

                        for file in os.listdir(subfolder_path):
                            if os.path.splitext(file)[0] == video_key:
                                file_path = os.path.join(subfolder_path, file)
                                value = pd.read_csv(file_path)
                                break
                        if value is not None:
                            break  # Break out of subfolder loop
                    if value is not None:
                        break  # Break out of folder_path loop

                if value is None:
                    continue  # Skip if file not found

                # Analyse crossings for presence of traffic equipment
                for unique_id, _ in crossings.items():
                    unique_id_indices = value.index[value['unique-id'] == unique_id]
                    if unique_id_indices.empty:
                        continue  # Skip if no occurrences

                    first_occurrence = unique_id_indices[0]
                    last_occurrence = unique_id_indices[-1]

                    yolo_ids = value.loc[first_occurrence:last_occurrence, 'yolo-id']

                    has_equipment = yolo_ids.isin([9, 11]).any()
                    lacks_equipment = not yolo_ids.isin([9, 11]).any()

                    if has_equipment:
                        count_with_equipment += 1
                    if lacks_equipment:
                        count_without_equipment += 1

                # Aggregate by city/condition
                crossings_with_traffic_equipment_city[location_key_city] = \
                    crossings_with_traffic_equipment_city.get(location_key_city, 0) + count_with_equipment
                crossings_without_traffic_equipment_city[location_key_city] = \
                    crossings_without_traffic_equipment_city.get(location_key_city, 0) + count_without_equipment

                # Aggregate by country/condition
                crossings_with_traffic_equipment_country[location_key_country] = \
                    crossings_with_traffic_equipment_country.get(location_key_country, 0) + count_with_equipment
                crossings_without_traffic_equipment_country[location_key_country] = \
                    crossings_without_traffic_equipment_country.get(location_key_country, 0) + count_without_equipment

        return (crossings_with_traffic_equipment_city,
                crossings_without_traffic_equipment_city,
                total_duration_by_city,
                crossings_with_traffic_equipment_country,
                crossings_without_traffic_equipment_country,
                total_duration_by_country)

    # TODO: combine methods for looking at crossing events with/without traffic lights
    @staticmethod
    def crossing_event_wt_traffic_light(df_mapping, data):
        """Plots traffic mortality rate vs percentage of crossing events without traffic light.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
            data (dict): Dictionary containing pedestrian crossing data.
        """
        var_exist, var_nt_exist, ratio = {}, {}, {}
        time_ = []

        counter_1, counter_2 = {}, {}

        # For a specific id of a person search for the first and last occurrence of that id and see if the traffic
        # light was present between it or not. Only getting those unique_id of the person who crosses the road.

        # Loop through each video data
        for key, df in tqdm(data.items(), total=len(data)):
            counter_exists, counter_nt_exists = 0, 0

            # Extract relevant information using the find_values function
            result = values_class.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                start = result[1]
                end = result[2]
                condition = result[3]
                city = result[4]
                lat = result[6]
                long = result[7]

                # Calculate the duration of the video
                duration = end - start
                time_.append(duration)

                for folder_path in common.get_configs('data'):
                    existing_subfolders = []
                    for subfolder in common.get_configs("sub_domain"):
                        subfolder_path = os.path.join(folder_path, subfolder)
                        if os.path.exists(subfolder_path):
                            existing_subfolders.append(subfolder)

                    # If none of the subfolders exist, print/log once
                    if not existing_subfolders:
                        logger.warning(f"None of the subfolders ('bbox', 'seg') exist in {folder_path}.")
                        continue

                    for subfolder in existing_subfolders:
                        subfolder_path = os.path.join(folder_path, subfolder)
                        for file in os.listdir(subfolder_path):
                            filename_no_ext = os.path.splitext(file)[0]
                            if filename_no_ext == key:
                                file_path = os.path.join(subfolder_path, file)
                                # Load the CSV
                                value = pd.read_csv(file_path)

                for id, time in df.items():
                    unique_id_indices = value.index[value['unique-id'] == id]
                    first_occurrence = unique_id_indices[0]
                    last_occurrence = unique_id_indices[-1]

                    # Check if yolo-id = 9 exists within the specified index range
                    yolo_id_9_exists = any(
                        value.loc[first_occurrence:last_occurrence, 'yolo-id'] == 9)
                    yolo_id_9_not_exists = not any(
                        value.loc[first_occurrence:last_occurrence, 'yolo-id'] == 9)

                    if yolo_id_9_exists:
                        counter_exists += 1
                    if yolo_id_9_not_exists:
                        counter_nt_exists += 1

                # Normalising the counters
                var_exist[key] = ((counter_exists * 60) / time_[-1])
                var_nt_exist[key] = ((counter_nt_exists * 60) / time_[-1])
                city_id_format = f'{city}_{lat}_{long}_{condition}'

                counter_1[city_id_format] = counter_1.get(city_id_format, 0) + var_exist[key]
                counter_2[city_id_format] = counter_2.get(city_id_format, 0) + var_nt_exist[key]

                if (counter_1[city_id_format] + counter_2[city_id_format]) == 0:
                    # Gives an error of division by 0
                    continue
                else:
                    if city_id_format in ratio:
                        ratio[city_id_format] = ((counter_2[city_id_format] * 100) /
                                                 (counter_1[city_id_format] +
                                                  counter_2[city_id_format]))
                        continue
                    # If already present, the array below will be filled multiple times
                    else:
                        ratio[city_id_format] = ((counter_2[city_id_format] * 100) /
                                                 (counter_1[city_id_format] +
                                                  counter_2[city_id_format]))
        return ratio

    @staticmethod
    def aggregate_by_iso3(df):
        """
        Aggregates a DataFrame by ISO3 country codes, applying specific aggregation rules.
        Drops unnecessary location-specific columns before processing.

        Parameters:
            df (pd.DataFrame): Original DataFrame with city-level traffic and demographic data.

        Returns:
            pd.DataFrame: Aggregated DataFrame grouped by ISO3 codes.
        """

        # Drop location-specific columns
        drop_columns = ['id', 'city', 'city_aka', 'state', 'lat', 'lon', 'gmp', 'population_city', 'traffic_index',
                        'upload_date', 'speed_crossing_day_city', 'speed_crossing_night_city',
                        'speed_crossing_day_night_city_avg', 'time_crossing_day_city',
                        'time_crossing_night_city', 'time_crossing_day_night_city_avg',
                        'with_trf_light_day_city', 'with_trf_light_night_city',
                        'without_trf_light_day_city', 'without_trf_light_night_city',
                        'crossing_detected_city', 'channel']

        static_columns = [
            'country', 'continent', 'population_country', 'traffic_mortality',
            'literacy_rate', 'avg_height', 'gini', 'med_age', 'speed_crossing_day_country',
            'speed_crossing_night_country', 'speed_crossing_day_night_country_avg',
            'time_crossing_day_country', 'time_crossing_night_country', 'time_crossing_day_night_country_avg',
            'with_trf_light_day_country', 'with_trf_light_night_country', 'without_trf_light_day_country',
            'without_trf_light_night_country', 'crossing_detected_country_all', 'crossing_detected_country_all_day',
            'crossing_detected_country_all_night', 'crossing_detected_country_day', 'crossing_detected_country',
            'crossing_detected_country_night'
            ]

        # Columns to merge as lists
        merge_columns = ['videos', 'time_of_day', 'start_time', 'end_time', 'vehicle_type', 'fps_list']

        sum_columns = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball',
            'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
            'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush', 'total_time', 'total_videos'
        ]

        # Only keep columns that exist in df
        drop_columns = [c for c in drop_columns if c in df.columns]
        static_columns = [c for c in static_columns if c in df.columns]
        merge_columns = [c for c in merge_columns if c in df.columns]
        sum_columns = [c for c in sum_columns if c in df.columns]

        # Drop location-specific columns
        df = df.drop(columns=drop_columns, errors='ignore')

        # Aggregation dictionary
        agg_dict = {
            **{col: 'first' for col in static_columns},
            **{col: (lambda x: list(x)) for col in merge_columns},
            **{col: 'sum' for col in sum_columns}
        }

        # Fix continent assignment if present
        if 'continent' in df.columns:
            continent_mode = (
                df.groupby('iso3')['continent']
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
                .reset_index()
                .rename(columns={'continent': 'continent_majority'})
            )
            df = df.drop('continent', axis=1)
            df = df.merge(continent_mode, on='iso3', how='left')
            df = df.rename(columns={'continent_majority': 'continent'})

        # Only keep columns in agg_dict that exist in df
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        # Aggregate
        df_grouped = df.groupby('iso3').agg(agg_dict).reset_index()

        return df_grouped

    @staticmethod
    def find_city_id(df, video_id, start_time):
        """
        Find the city identifier (row 'id') associated with a given video ID and start time.

        This function iterates through a DataFrame where each row may reference multiple videos and
        corresponding start times (stored as lists). It returns the 'id' value from the row where both
        the video ID and the exact start time match.

        Args:
            df (pd.DataFrame): DataFrame containing at least:
                               - 'videos': a string representing a list of video IDs.
                               - 'start_time': a string representing a list of lists of start times.
                               - 'id': unique identifier for each row (e.g., city or condition group).
            video_id (str): The video filename or identifier to search for.
            start_time (float or int): The specific start time to match within the corresponding list.

        Returns:
            The value of the 'id' field in the matching row, or None if no match is found.
        """
        logger.debug(f"{video_id}: looking for city, start_time={start_time}.")

        for _, row in df.iterrows():
            # Convert comma-separated string to list of video IDs
            videos = re.findall(r"[\w-]+", row["videos"])
            # Parse the stringified list of lists
            start_times = ast.literal_eval(row["start_time"])

            if video_id in videos:
                index = videos.index(video_id)

                # Check if the provided start_time exists in the relevant sublist
                if start_time in start_times[index]:
                    return row["id"]

        # No match found
        return None

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
                                result = values_class.find_values_with_video_id(df_mapping, video_start_time)

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

    @staticmethod
    def get_coordinates(city, state, country):
        """
        Retrieve geographic coordinates (latitude and longitude) for a given city, state, and country.

        The function uses the Nominatim geocoding service from the geopy library. It first constructs a location
        query string based on the provided city, optional state, and country. A unique user-agent is generated
        for each call to avoid getting blocked by the server.

        Args:
            city (str): Name of the city.
            state (str or None): Name of the state or province. Optional; can be None or 'nan'.
            country (str): Name of the country.

        Returns:
            tuple: A tuple (latitude, longitude) if geocoding is successful, otherwise (None, None).

        Exceptions:
            - Logs and handles `GeocoderTimedOut` if the geocoding request times out.
            - Logs and handles `GeocoderUnavailable` if the geocoding server is unreachable.
        """
        # Generate a unique user agent with the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        user_agent = f"my_geocoding_script_{current_time}"

        # Initialise the Nominatim geocoder with the unique user agent
        geolocator = Nominatim(user_agent=user_agent)

        try:
            # Form the query string depending on whether a valid state value is provided
            if state and str(state).lower() != 'nan':
                location_query = f"{city}, {state}, {country}"  # Combine city, state and country
            else:
                location_query = f"{city}, {country}"  # Combine city and country
            location = geolocator.geocode(location_query, timeout=2)  # type: ignore # Set a 2-second timeout

            if location:
                return location.latitude, location.longitude  # type: ignore
            else:
                logger.error(f"Failed to geocode {location_query}")
                return None, None  # Return None if city is not found

        except GeocoderTimedOut:
            # Handle timeout errors when the request takes too long
            logger.error(f"Geocoding timed out for {location_query}.")
        except GeocoderUnavailable:
            # Handle cases where the geocoding service is not available
            logger.error(f"Geocoding server could not be reached for {location_query}.")
            return None, None  # Return None if city is not found

    def add_speed_and_time_to_mapping(self, df_mapping, avg_speed_city, avg_time_city, avg_speed_country,
                                      avg_time_country, pedestrian_cross_city, pedestrian_cross_country,
                                      threshold=common.get_configs("min_crossing_detect")):
        """
        Adds city/country-level average speeds and/or times (day/night) to df_mapping DataFrame,
        depending on which dicts are provided. Missing columns are created and initialised with NaN.
        For country-level data, values are only added if pedestrian_cross_country[country_cond] < value.
        """
        configs = []
        if avg_speed_city is not None:
            configs.append(dict(
                label='city',
                avg_dict=avg_speed_city,
                value_type='speed',
                col_prefix='speed_crossing',
                key_parts=['city', 'lat', 'long', 'time_of_day'],
                get_state=True
            ))
        if avg_time_city is not None:
            configs.append(dict(
                label='city',
                avg_dict=avg_time_city,
                value_type='time',
                col_prefix='time_crossing',
                key_parts=['city', 'lat', 'long', 'time_of_day'],
                get_state=True
            ))
        if avg_speed_country is not None:
            configs.append(dict(
                label='country',
                avg_dict=avg_speed_country,
                value_type='speed',
                col_prefix='speed_crossing',
                key_parts=['country', 'time_of_day'],
                get_state=False
            ))
        if avg_time_country is not None:
            configs.append(dict(
                label='country',
                avg_dict=avg_time_country,
                value_type='time',
                col_prefix='time_crossing',
                key_parts=['country', 'time_of_day'],
                get_state=False
            ))

        for cfg in configs:
            label = cfg['label']
            avg_dict = cfg['avg_dict']
            col_prefix = cfg['col_prefix']
            get_state = cfg['get_state']  # noqa:F841

            # Prepare column names
            day_col = f"{col_prefix}_day_{label}"
            night_col = f"{col_prefix}_night_{label}"
            avg_col = f"{col_prefix}_day_night_{label}_avg"

            # Ensure columns exist and are initialised to np.nan
            for col in [day_col, night_col, avg_col]:
                if col not in df_mapping.columns:
                    df_mapping[col] = np.nan

            for key, value in tqdm(avg_dict.items(), desc=f"{label.capitalize()} {cfg['value_type'].capitalize()}s",
                                   total=len(avg_dict)):
                parts = key.split("_")
                if label == 'city':
                    city, lat, _, time_of_day = parts[0], parts[1], parts[2], int(parts[3])
                    state = values_class.get_value(df_mapping, "city", city, "lat", lat, "state")
                    mask = (
                        (df_mapping["city"] == city) &
                        ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state)))
                    )
                else:  # country
                    # Parse country key
                    country, time_of_day = parts[0], int(parts[1])

                    # Check pedestrian_cross_country condition
                    cond_key = f"{country}_{time_of_day}"
                    cross_value = pedestrian_cross_country.get(cond_key, 0)
                    if cross_value <= threshold:
                        continue  # Skip if condition is not satisfied

                    mask = (df_mapping["country"] == country)
                if not time_of_day:
                    df_mapping.loc[mask, day_col] = float(value)
                else:
                    df_mapping.loc[mask, night_col] = float(value)

            # Calculate overall average column for each type
            df_mapping[avg_col] = np.where(
                (df_mapping[day_col] > 0) & (df_mapping[night_col] > 0),
                df_mapping[[day_col, night_col]].mean(axis=1),
                np.where(
                    df_mapping[day_col] > 0, df_mapping[day_col],
                    np.where(df_mapping[night_col] > 0, df_mapping[night_col], np.nan)
                )
            )

        return df_mapping


analysis_class = Analysis()

# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")

    if os.path.exists(file_results) and not common.get_configs('always_analyse'):
        # Load the data from the pickle file
        with open(file_results, 'rb') as file:
            (data,                                          # 0
             person_counter,                                # 1
             bicycle_counter,                               # 2
             car_counter,                                   # 3
             motorcycle_counter,                            # 4
             bus_counter,                                   # 5
             truck_counter,                                 # 6
             cellphone_counter,                             # 7
             traffic_light_counter,                         # 8
             stop_sign_counter,                             # 9
             pedestrian_cross_city,                         # 10
             pedestrian_crossing_count,                     # 11
             person_city,                                   # 12
             bicycle_city,                                  # 13
             car_city,                                      # 14
             motorcycle_city,                               # 15
             bus_city,                                      # 16
             truck_city,                                    # 17
             cross_evnt_city,                               # 18
             vehicle_city,                                  # 19
             cellphone_city,                                # 20
             traffic_sign_city,                             # 21
             all_speed,                                     # 22
             all_time,                                      # 23
             avg_time_city,                                 # 24
             avg_speed_city,                                # 25
             df_mapping,                                    # 26
             avg_speed_country,                             # 27
             avg_time_country,                              # 28
             crossings_with_traffic_equipment_city,         # 29
             crossings_without_traffic_equipment_city,      # 30
             crossings_with_traffic_equipment_country,      # 31
             crossings_without_traffic_equipment_country,   # 32
             min_max_speed,                                 # 33
             min_max_time,                                  # 34
             pedestrian_cross_country,                      # 35
             all_speed_city,                                # 36
             all_time_city,                                 # 37
             all_speed_country,                             # 38
             all_time_country,                              # 39
             df_mapping_raw,                                # 40
             pedestrian_cross_city_all,                     # 41
             pedestrian_cross_country_all                   # 42
             ) = pickle.load(file)

        logger.info("Loaded analysis results from pickle file.")
    else:
        # Store the mapping file
        df_mapping = pd.read_csv(common.get_configs("mapping"))

        # Produce map with all data
        df = df_mapping.copy()  # copy df to manipulate for output
        df['state'] = df['state'].fillna('NA')  # Set state to NA

        # Sort by continent and city, both in ascending order
        df = df.sort_values(by=["continent", "city"], ascending=[True, True])

        # Count of videos
        df['video_count'] = df['videos'].apply(lambda x: len(x.strip('[]').split(',')) if pd.notna(x) else 0)

        # Total amount of seconds in segments
        def flatten(lst):
            """Flattens nested lists like [[1, 2], [3, 4]] -> [1, 2, 3, 4]"""
            return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]

        def compute_total_time(row):
            try:
                start_times = flatten(ast.literal_eval(row['start_time']))
                end_times = flatten(ast.literal_eval(row['end_time']))
                return sum(e - s for s, e in zip(start_times, end_times))
            except Exception as e:
                logger.error(f"Error in row {row['id']}: {e}")
                return 0

        df['total_time'] = df.apply(compute_total_time, axis=1)

        # Data to avoid showing on hover in scatter plots
        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'vehicle_type', 'channel']
        hover_data = list(set(df.columns) - set(columns_remove))

        # Sort by continent and city, both in ascending order
        df = df.sort_values(by=["continent", "country"], ascending=[True, True])

        # maps with all data
        plots_class.mapbox_map(df=df, hover_data=hover_data, file_name='mapbox_map_all')
        plots_class.mapbox_map(df=df,
                               hover_data=hover_data,
                               density_col='population_city',
                               density_radius=10,
                               file_name='mapbox_map_all_pop')
        plots_class.mapbox_map(df=df,
                               hover_data=hover_data,
                               density_col='video_count',
                               density_radius=10,
                               file_name='mapbox_map_all_videos')
        plots_class.mapbox_map(df=df,
                               hover_data=hover_data,
                               density_col='total_time',
                               density_radius=10,
                               file_name='mapbox_map_all_time')

        # Sort by continent and city, both in ascending order
        df = df.sort_values(by=["country", "city"], ascending=[True, True])

        # scatter plot with number of videos over total time
        plots_class.scatter(df=df,
                            x="total_time",
                            y="video_count",
                            color="country",
                            # text="city",
                            xaxis_title='Total time of footage (s)',
                            yaxis_title='Number of videos',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            # legend_x=0.01,
                            # legend_y=1.0,
                            label_distance_factor=5.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None,  # type: ignore
                            file_name='scatter_all_total_time-video_count')  # type: ignore

        total_duration = values_class.calculate_total_seconds(df_mapping)

        # Displays values before applying filters
        logger.info(f"Duration of videos in seconds: {total_duration}, in minutes: {total_duration/60:.2f}, in " +
                    f"hours: {total_duration/60/60:.2f} before filtering.")
        logger.info("Total number of videos before filtering: {}.", values_class.calculate_total_videos(df_mapping))

        country, number = Analysis.get_unique_values(df_mapping, "iso3")
        logger.info("Total number of countries and territories before filtering: {}.", number)

        city, number = Analysis.get_unique_values(df_mapping, "city")
        logger.info("Total number of cities before filtering: {}.", number)

        # Limit countries if required
        countries_include = common.get_configs("countries_analyse")
        if countries_include:
            df_mapping = df_mapping[df_mapping["iso3"].isin(common.get_configs("countries_analyse"))]

        # Make a dict for all columns
        city_country_cols = {
            # Object columns
            'person': 0, 'bicycle': 0, 'car': 0, 'motorcycle': 0, 'airplane': 0, 'bus': 0, 'train': 0,
            'truck': 0, 'boat': 0, 'traffic_light': 0, 'fire_hydrant': 0, 'stop_sign': 0, 'parking_meter': 0,
            'bench': 0, 'bird': 0, 'cat': 0, 'dog': 0, 'horse': 0, 'sheep': 0, 'cow': 0, 'elephant': 0, 'bear': 0,
            'zebra': 0, 'giraffe': 0, 'backpack': 0, 'umbrella': 0, 'handbag': 0, 'tie': 0, 'suitcase': 0,
            'frisbee': 0, 'skis': 0, 'snowboard': 0, 'sports_ball': 0, 'kite': 0, 'baseball_bat': 0,
            'baseball_glove': 0, 'skateboard': 0, 'surfboard': 0, 'tennis_racket': 0, 'bottle': 0, 'wine_glass': 0,
            'cup': 0, 'fork': 0, 'knife': 0, 'spoon': 0, 'bowl': 0, 'banana': 0, 'apple': 0, 'sandwich': 0,
            'orange': 0, 'broccoli': 0, 'carrot': 0, 'hot_dog': 0, 'pizza': 0, 'donut': 0, 'cake': 0, 'chair': 0,
            'couch': 0, 'potted_plant': 0, 'bed': 0, 'dining_table': 0, 'toilet': 0, 'tv': 0, 'laptop': 0,
            'mouse': 0, 'remote': 0, 'keyboard': 0, 'cellphone': 0, 'microwave': 0, 'oven': 0, 'toaster': 0,
            'sink': 0, 'refrigerator': 0, 'book': 0, 'clock': 0, 'vase': 0, 'scissors': 0, 'teddy_bear': 0,
            'hair_drier': 0, 'toothbrush': 0,

            'total_time': 0,
            'total_crossing_detect': 0,

            # City-level columns
            'speed_crossing_day_city': math.nan,
            'speed_crossing_night_city': math.nan,
            'speed_crossing_day_night_city_avg': math.nan,
            'time_crossing_day_city': math.nan,
            'time_crossing_night_city': math.nan,
            'time_crossing_day_night_city_avg': math.nan,
            'with_trf_light_day_city': 0.0,
            'with_trf_light_night_city': 0.0,
            'without_trf_light_day_city': 0.0,
            'without_trf_light_night_city': 0.0,
            'crossing_detected_city': 0,
            'crossing_detected_city_day': 0,
            'crossing_detected_city_night': 0,
            'crossing_detected_city_all': 0,
            'crossing_detected_city_all_day': 0,
            'crossing_detected_city_all_night': 0,

            # Country-level columns
            'speed_crossing_day_country': math.nan,
            'speed_crossing_night_country': math.nan,
            'speed_crossing_day_night_country_avg': math.nan,
            'time_crossing_day_country': math.nan,
            'time_crossing_night_country': math.nan,
            'time_crossing_day_night_country_avg': math.nan,
            'with_trf_light_day_country': 0.0,
            'with_trf_light_night_country': 0.0,
            'without_trf_light_day_country': 0.0,
            'without_trf_light_night_country': 0.0,
            'crossing_detected_country': 0,
            'crossing_detected_country_day': 0,
            'crossing_detected_country_night': 0,
            'crossing_detected_country_all': 0,
            'crossing_detected_country_all_day': 0,
            'crossing_detected_country_all_night': 0,
        }

        # Efficiently add all columns at once
        cols_df = pd.DataFrame([city_country_cols] * len(df_mapping), index=df_mapping.index)

        df_mapping = pd.concat([df_mapping, cols_df], axis=1)

        all_speed = {}
        all_time = {}

        logger.info("Processing csv files.")
        pedestrian_crossing_count, data = {}, {}
        pedestrian_crossing_count_all = {}

        for folder_path in common.get_configs("data"):  # Iterable[str]
            if not os.path.exists(folder_path):
                logger.warning(f"Folder does not exist: {folder_path}.")
                continue

            found_any = False

            for subfolder in common.get_configs("sub_domain"):
                subfolder_path = os.path.join(folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    continue

                found_any = True

                for file_name in tqdm(os.listdir(subfolder_path), desc=f"Processing files in {subfolder_path}"):
                    filtered: Optional[str] = analysis_class.filter_csv_files(
                        file=file_name, df_mapping=df_mapping
                    )
                    if filtered is None:
                        continue

                    # Ensure "file" is always a string
                    file_str: str = os.fspath(filtered)  # converts PathLike to str safely

                    if file_str in misc_files:
                        continue

                    filename_no_ext = os.path.splitext(file_str)[0]
                    logger.debug(f"{filename_no_ext}: fetching values.")

                    file_path = os.path.join(subfolder_path, file_str)
                    df = pd.read_csv(file_path)

                    # After reading the file, clean up the filename
                    base_name = tools_class.clean_csv_filename(file_str)
                    filename_no_ext = os.path.splitext(base_name)[0]  # Remove extension

                    try:
                        video_id, start_index, fps = filename_no_ext.rsplit("_", 2)  # split to extract id and index
                    except ValueError:
                        logger.warning(f"Unexpected filename format: {filename_no_ext}")
                        continue

                    video_city_id = Analysis.find_city_id(df_mapping, video_id, int(start_index))
                    video_city = df_mapping.loc[df_mapping["id"] == video_city_id, "city"].values[0]  # type: ignore # noqa: E501
                    video_state = df_mapping.loc[df_mapping["id"] == video_city_id, "state"].values[0]  # type: ignore # noqa: E501
                    video_country = df_mapping.loc[df_mapping["id"] == video_city_id, "country"].values[0]  # type: ignore # noqa: E501
                    logger.debug(f"{file_str}: found values {video_city}, {video_state}, {video_country}.")

                    # Get the number of number and unique id of the object crossing the road
                    # ids give the unique of the person who cross the road after applying the filter, while
                    # all_ids gives every unique_id of the person who crosses the road
                    ids, all_ids = algorithms_class.pedestrian_crossing(df,
                                                                        filename_no_ext,
                                                                        df_mapping,
                                                                        common.get_configs("boundary_left"),
                                                                        common.get_configs("boundary_right"),
                                                                        person_id=0)
                    # Saving it in a dictionary in: {video-id_time: count, ids}
                    pedestrian_crossing_count[filename_no_ext] = {"ids": ids}
                    pedestrian_crossing_count_all[filename_no_ext] = {"ids": all_ids}
                    # Saves the time to cross in form {name_time: {id(s): time(s)}}
                    temp_data = algorithms_class.time_to_cross(df,
                                                               pedestrian_crossing_count[filename_no_ext]["ids"],
                                                               filename_no_ext,
                                                               df_mapping)
                    data[filename_no_ext] = temp_data
                    # List of all 80 class names in COCO order
                    coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                                    'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench',
                                    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                                    'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
                                    'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup',
                                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                                    'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                    'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                                    'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink',
                                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear',
                                    'hair_drier', 'toothbrush']
                    # --- Ensure all needed columns exist and are integer type ---
                    for class_name in coco_classes:
                        if class_name not in df_mapping.columns:
                            df_mapping[class_name] = 0
                        df_mapping[class_name] = pd.to_numeric(df_mapping[class_name],
                                                               errors='coerce').fillna(0).astype(int)
                    # --- Count unique objects per yolo-id ---
                    object_counts = (
                        df.drop_duplicates(['yolo-id', 'unique-id'])['yolo-id']
                        .value_counts().sort_index()
                    )
                    counters = {class_name: int(object_counts.get(i, 0)) for i,
                                class_name in enumerate(coco_classes)}
                    # --- Update df_mapping for the given video_city_id ---
                    for class_name in coco_classes:
                        df_mapping.loc[df_mapping["id"] == video_city_id, class_name] += counters[class_name]  # type: ignore  # noqa: E501
                    # Add duration of segment
                    time_video = analysis_class.get_duration(df_mapping, video_id, int(start_index))
                    df_mapping.loc[df_mapping["id"] == video_city_id, "total_time"] += time_video  # type: ignore
                    # Add total crossing detected
                    df_mapping.loc[df_mapping["id"] == video_city_id, "total_crossing_detect"] += len(ids)  # type: ignore  # noqa: E501
                    # Aggregated values
                    speed_value = algorithms_class.calculate_speed_of_crossing(df_mapping,
                                                                               df,
                                                                               {filename_no_ext: temp_data})
                    if speed_value is not None:
                        for outer_key, inner_dict in speed_value.items():
                            if outer_key not in all_speed:
                                all_speed[outer_key] = inner_dict
                            else:
                                all_speed[outer_key].update(inner_dict)
                    time_value = algorithms_class.time_to_start_cross(df_mapping,
                                                                      df,
                                                                      {filename_no_ext: temp_data})
                    if time_value is not None:
                        for outer_key, inner_dict in time_value.items():
                            if outer_key not in all_time:
                                all_time[outer_key] = inner_dict
                            else:
                                all_time[outer_key].update(inner_dict)

        person_counter = df_mapping['person'].sum()
        bicycle_counter = df_mapping['bicycle'].sum()
        car_counter = df_mapping['car'].sum()
        motorcycle_counter = df_mapping['motorcycle'].sum()
        bus_counter = df_mapping['bus'].sum()
        truck_counter = df_mapping['truck'].sum()
        cellphone_counter = df_mapping['cellphone'].sum()
        traffic_light_counter = df_mapping['traffic_light'].sum()
        stop_sign_counter = df_mapping['stop_sign'].sum()

        # Record the average speed and time of crossing on country basis
        avg_speed_country, all_speed_country = algorithms_class.avg_speed_of_crossing_country(df_mapping, all_speed)

        # Output in real world seconds
        avg_time_country, all_time_country = algorithms_class.avg_time_to_start_cross_country(df_mapping, all_time)

        # Record the average speed and time of crossing on city basis
        avg_speed_city, all_speed_city = algorithms_class.avg_speed_of_crossing_city(df_mapping, all_speed)
        avg_time_city, all_time_city = algorithms_class.avg_time_to_start_cross_city(df_mapping, all_time)

        # Kill the program if there is no data to analyse
        if len(avg_time_city) == 0 or len(avg_speed_city) == 0:
            logger.error("No speed and time data to analyse.")
            exit()

        logger.info("Calculating counts of detected traffic signs.")
        traffic_sign_city = analysis_class.calculate_traffic_signs(df_mapping)
        logger.info("Calculating counts of detected mobile phones.")
        cellphone_city = Analysis.calculate_cellphones(df_mapping)
        logger.info("Calculating counts of detected vehicles.")
        vehicle_city = analysis_class.calculate_traffic(df_mapping, motorcycle=1, car=1, bus=1, truck=1)
        logger.info("Calculating counts of detected bicycles.")
        bicycle_city = analysis_class.calculate_traffic(df_mapping, bicycle=1)
        logger.info("Calculating counts of detected cars (subset of vehicles).")
        car_city = analysis_class.calculate_traffic(df_mapping, car=1)
        logger.info("Calculating counts of detected motorcycles (subset of vehicles).")
        motorcycle_city = analysis_class.calculate_traffic(df_mapping, motorcycle=1)
        logger.info("Calculating counts of detected buses (subset of vehicles).")
        bus_city = analysis_class.calculate_traffic(df_mapping, bus=1)
        logger.info("Calculating counts of detected trucks (subset of vehicles).")
        truck_city = analysis_class.calculate_traffic(df_mapping, truck=1)
        logger.info("Calculating counts of detected persons.")
        person_city = analysis_class.calculate_traffic(df_mapping, person=1)
        logger.info("Calculating counts of detected crossing events with traffic lights.")
        cross_evnt_city = Analysis.crossing_event_wt_traffic_light(df_mapping, data)
        logger.info("Calculating counts of crossing events in cities.")
        pedestrian_cross_city = values_class.pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping)
        pedestrian_cross_city_all = values_class.pedestrian_cross_per_city(pedestrian_crossing_count_all, df_mapping)
        logger.info("Calculating counts of crossing events in countries.")
        pedestrian_cross_country = values_class.pedestrian_cross_per_country(pedestrian_cross_city, df_mapping)
        pedestrian_cross_country_all = values_class.pedestrian_cross_per_country(pedestrian_cross_city_all, df_mapping)

        # Jaywalking data
        logger.info("Calculating parameters for detection of jaywalking.")

        (crossings_with_traffic_equipment_city, crossings_without_traffic_equipment_city,
         total_duration_by_city, crossings_with_traffic_equipment_country, crossings_without_traffic_equipment_country,
         total_duration_by_country) = Analysis.crossing_event_with_traffic_equipment(df_mapping, data)

        # ----------------------------------------------------------------------
        # Add city-level crossing counts for with and without traffic equipment
        # ----------------------------------------------------------------------

        for key, value in crossings_with_traffic_equipment_city.items():
            parts = key.split("_")
            city = parts[0]
            lat = parts[1]
            long = parts[2]
            time_of_day = int(parts[3])  # 0 = day, 1 = night

            # Optional: Extract state if available
            state = df_mapping.loc[df_mapping["city"] == city,
                                   "state"].iloc[0] if "state" in df_mapping.columns else None  # type: ignore

            colname = "with_trf_light_day_city" if not time_of_day else "with_trf_light_night_city"

            df_mapping.loc[
                (df_mapping["city"] == city) &
                ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                colname
            ] = float(value)

        for key, value in crossings_without_traffic_equipment_city.items():
            parts = key.split("_")
            city = parts[0]
            lat = parts[1]
            long = parts[2]
            time_of_day = int(parts[3])

            # Optional: Extract state if available
            state = df_mapping.loc[df_mapping["city"] == city,
                                   "state"].iloc[0] if "state" in df_mapping.columns else None  # type: ignore

            colname = "without_trf_light_day_city" if not time_of_day else "without_trf_light_night_city"

            df_mapping.loc[
                (df_mapping["city"] == city) &
                ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                colname
            ] = float(value)

        # ----------------------------------------------------------------------
        # Add country-level crossing counts for with and without traffic equipment
        # ----------------------------------------------------------------------

        for key, value in crossings_with_traffic_equipment_country.items():
            parts = key.split("_")
            country = parts[0]
            time_of_day = int(parts[1])

            colname = "with_trf_light_day_country" if not time_of_day else "with_trf_light_night_country"

            df_mapping.loc[
                (df_mapping["country"] == country),
                colname
            ] = float(value)

        for key, value in crossings_without_traffic_equipment_country.items():
            parts = key.split("_")
            country = parts[0]
            time_of_day = int(parts[1])

            colname = "without_trf_light_day_country" if not time_of_day else "without_trf_light_night_country"

            df_mapping.loc[
                (df_mapping["country"] == country),
                colname
            ] = float(value)

        # ---------------------------------------
        # Add city-level crossing counts detected
        # ---------------------------------------
        for city_long_lat_cond, value in pedestrian_cross_city.items():
            city, lat, long, cond = city_long_lat_cond.split('_')
            lat = float(lat)  # lat column is float

            # Set the correct column name based on condition
            if cond == "0":
                target_column = "crossing_detected_city_day"
            elif cond == "1":
                target_column = "crossing_detected_city_night"
            else:
                continue  # skip if cond is not recognised

            # Set the value in the right place
            df_mapping.loc[
                (df_mapping["city"] == city) & (df_mapping["lat"] == lat),
                target_column
            ] = float(value)

        for city_long_lat_cond, value in pedestrian_cross_city_all.items():
            city, lat, long, cond = city_long_lat_cond.split('_')
            lat = float(lat)  # if your lat column is float
            # Set the correct column name based on condition
            if cond == "0":
                target_column = "crossing_detected_city_all_day"
            elif cond == "1":
                target_column = "crossing_detected_city_all_night"
            else:
                continue  # skip if cond is not recognized
            # Set the value in the right place
            df_mapping.loc[
                (df_mapping["city"] == city) & (df_mapping["lat"] == lat),
                target_column
            ] = float(value)

        df_mapping["crossing_detected_city"] = (
            df_mapping["crossing_detected_city_day"].fillna(0)
            + df_mapping["crossing_detected_city_night"].fillna(0)
        )

        df_mapping["crossing_detected_city_all"] = (
            df_mapping["crossing_detected_city_all_day"].fillna(0)
            + df_mapping["crossing_detected_city_all_night"].fillna(0)
        )

        # ---------------------------------------
        # Add country-level crossing counts detected
        # ---------------------------------------
        for country_cond, value in pedestrian_cross_country.items():
            country, cond = country_cond.split('_')
            # Set the correct column name based on condition
            if cond == "0":
                target_column = "crossing_detected_country_day"
            elif cond == "1":
                target_column = "crossing_detected_country_night"
            else:
                continue  # skip if cond is not recognized
            # Set the value in the right place
            df_mapping.loc[
                (df_mapping["country"] == country),
                target_column
            ] = float(value)

        for country_cond, value in pedestrian_cross_country_all.items():
            country, cond = country_cond.split('_')
            # Set the correct column name based on condition
            if cond == "0":
                target_column = "crossing_detected_country_all_day"
            elif cond == "1":
                target_column = "crossing_detected_country_all_night"
            else:
                continue  # skip if cond is not recognized
            # Set the value in the right place
            df_mapping.loc[
                (df_mapping["country"] == country),
                target_column
            ] = float(value)

        df_mapping["crossing_detected_country"] = (
            df_mapping["crossing_detected_country_day"].fillna(0)
            + df_mapping["crossing_detected_country_night"].fillna(0)
        )

        df_mapping["crossing_detected_country_all"] = (
            df_mapping["crossing_detected_country_all_day"].fillna(0)
            + df_mapping["crossing_detected_country_all_night"].fillna(0)
        )

        # Add column with count of videos
        df_mapping["total_videos"] = df_mapping["videos"].apply(lambda x: len(x.strip("[]").split(",")) if x.strip("[]") else 0)  # noqa: E501

        # Get lat and lon for cities
        logger.info("Fetching lat and lon coordinates for cities.")
        for index, row in tqdm(df_mapping.iterrows(), total=len(df_mapping)):
            if pd.isna(row["lat"]) or pd.isna(row["lon"]):
                lat, lon = Analysis.get_coordinates(row["city"],
                                                    row["state"],
                                                    common.correct_country(row["country"]))  # type: ignore
                df_mapping.at[index, 'lat'] = lat
                df_mapping.at[index, 'lon'] = lon

        # Save the raw file for further investigation
        df_mapping_raw = df_mapping.copy()

        df_mapping_raw.drop(['gmp', 'population_city', 'population_country', 'traffic_mortality',
                             'literacy_rate', 'avg_height', 'med_age', 'gini', 'traffic_index', 'videos',
                             'time_of_day', 'start_time', 'end_time', 'vehicle_type', 'upload_date',
                             ], axis=1, inplace=True)

        df_mapping_raw['channel'] = df_mapping_raw['channel'].apply(tools_class.count_unique_channels)
        df_mapping_raw.to_csv(os.path.join(common.output_dir, "mapping_city_raw.csv"))

        # Get the population threshold from the configuration
        population_threshold = common.get_configs("population_threshold")

        # Get the minimum percentage of country population from the configuration
        min_percentage = common.get_configs("min_city_population_percentage")

        # Convert 'population_city' to numeric (force errors to NaN)
        df_mapping["population_city"] = pd.to_numeric(df_mapping["population_city"], errors='coerce')

        # Filter df_mapping to include cities that meet either of the following criteria:
        # 1. The city's population is greater than the threshold
        # 2. The city's population is at least the minimum percentage of the country's population
        df_mapping = df_mapping[
            (df_mapping["population_city"] >= population_threshold) |  # Condition 1
            (df_mapping["population_city"] >= min_percentage * df_mapping["population_country"])  # Condition 2
        ]

        # Remove the rows of the cities where the footage recorded is less than threshold
        df_mapping = values_class.remove_columns_below_threshold(df_mapping, common.get_configs("footage_threshold"))

        # Limit countries if required
        countries_include = common.get_configs("countries_analyse")
        if countries_include:
            df_mapping = df_mapping[df_mapping["iso3"].isin(common.get_configs("countries_analyse"))]

        total_duration = values_class.calculate_total_seconds(df_mapping)

        # Displays values after applying filters
        logger.info(f"Duration of videos in seconds after filtering: {total_duration}, in" +
                    f" minutes after filtering: {total_duration/60:.2f}, in " +
                    f"hours: {total_duration/60/60:.2f}.")

        logger.info("Total number of videos after filtering: {}.", values_class.calculate_total_videos(df_mapping))

        country, number = Analysis.get_unique_values(df_mapping, "iso3")
        logger.info("Total number of countries and territories after filtering: {}.", number)

        city, number = Analysis.get_unique_values(df_mapping, "city")
        logger.info("Total number of cities after filtering: {}.", number)

        df_mapping = analysis_class.add_speed_and_time_to_mapping(df_mapping=df_mapping,
                                                                  avg_speed_city=avg_speed_city,
                                                                  avg_speed_country=avg_speed_country,
                                                                  avg_time_city=avg_time_city,
                                                                  avg_time_country=avg_time_country,
                                                                  pedestrian_cross_city=pedestrian_cross_city,
                                                                  pedestrian_cross_country=pedestrian_cross_country)

        min_max_speed = analysis_class.get_duration_segment(all_speed, df_mapping, name="speed", duration=None)
        min_max_time = analysis_class.get_duration_segment(all_time, df_mapping, name="time", duration=None)

        # Save the results to a pickle file
        logger.info("Saving results to a pickle file {}.", file_results)
        with open(file_results, 'wb') as file:
            pickle.dump((data,                                              # 0
                         person_counter,                                    # 1
                         bicycle_counter,                                   # 2
                         car_counter,                                       # 3
                         motorcycle_counter,                                # 4
                         bus_counter,                                       # 5
                         truck_counter,                                     # 6
                         cellphone_counter,                                 # 7
                         traffic_light_counter,                             # 8
                         stop_sign_counter,                                 # 9
                         pedestrian_cross_city,                             # 10
                         pedestrian_crossing_count,                         # 11
                         person_city,                                       # 12
                         bicycle_city,                                      # 13
                         car_city,                                          # 14
                         motorcycle_city,                                   # 15
                         bus_city,                                          # 16
                         truck_city,                                        # 17
                         cross_evnt_city,                                   # 18
                         vehicle_city,                                      # 19
                         cellphone_city,                                    # 20
                         traffic_sign_city,                                 # 21
                         all_speed,                                         # 22
                         all_time,                                          # 23
                         avg_time_city,                                     # 24
                         avg_speed_city,                                    # 25
                         df_mapping,                                        # 26
                         avg_speed_country,                                 # 27
                         avg_time_country,                                  # 28
                         crossings_with_traffic_equipment_city,             # 29
                         crossings_without_traffic_equipment_city,          # 30
                         crossings_with_traffic_equipment_country,          # 31
                         crossings_without_traffic_equipment_country,       # 32
                         min_max_speed,                                     # 33
                         min_max_time,                                      # 34
                         pedestrian_cross_country,                          # 35
                         all_speed_city,                                    # 36
                         all_time_city,                                     # 37
                         all_speed_country,                                 # 38
                         all_time_country,                                  # 39
                         df_mapping_raw,                                    # 40
                         pedestrian_cross_city_all,                         # 41
                         pedestrian_cross_country_all),                     # 42
                        file)

        logger.info("Analysis results saved to pickle file.")

    # Set index as ID
    df_mapping = df_mapping.set_index("id", drop=False)

    # --- Check if reanalysis of speed is required ---
    if common.get_configs("reanalyse_speed"):
        # Compute average speed for each country using mapping and speed data
        avg_speed_country = algorithms_class.avg_speed_of_crossing_country(df_mapping, all_speed)
        # Compute average speed for each city using speed data
        avg_speed_city = algorithms_class.avg_speed_of_crossing_city(df_mapping, all_speed)

        # Add computed speed values to the main mapping dataframe
        df_mapping = analysis_class.add_speed_and_time_to_mapping(
            df_mapping=df_mapping,
            avg_speed_city=avg_speed_city,
            avg_time_city=None,
            avg_speed_country=avg_speed_country,
            avg_time_country=None,
            pedestrian_cross_city=pedestrian_cross_city,
            pedestrian_cross_country=pedestrian_cross_country
        )

        # --- Update avg speed values in the pickle file ---
        with open(file_results, 'rb') as file:
            results = pickle.load(file)  # Load existing results

        results_list = list(results)
        results_list[25] = avg_speed_city     # Update city speed
        results_list[27] = avg_speed_country  # Update country speed
        results_list[26] = df_mapping         # Update mapping

        with open(file_results, 'wb') as file:
            pickle.dump(tuple(results_list), file)  # Save updated results
        logger.info("Updated speed values in the pickle file.")

    # --- Check if reanalysis of waiting time is required ---
    if common.get_configs("reanalyse_waiting_time"):
        # Compute average waiting time to start crossing for each country
        avg_time_country = algorithms_class.avg_time_to_start_cross_country(df_mapping, all_speed)
        # Compute average waiting time to start crossing for each city
        avg_time_city = algorithms_class.avg_time_to_start_cross_city(df_mapping, all_time)

        # Add computed time values to the main mapping dataframe
        df_mapping = analysis_class.add_speed_and_time_to_mapping(
            df_mapping=df_mapping,
            avg_time_city=avg_time_city,
            avg_speed_city=avg_speed_city,
            avg_time_country=avg_time_country,
            avg_speed_country=avg_speed_country,
            pedestrian_cross_city=pedestrian_cross_city,
            pedestrian_cross_country=pedestrian_cross_country
        )

        # --- Update avg time values in the pickle file ---
        with open(file_results, 'rb') as file:
            results = pickle.load(file)  # Load existing results

        results_list = list(results)
        results_list[24] = avg_time_city     # Update city waiting time
        results_list[28] = avg_time_country  # Update country waiting time
        results_list[26] = df_mapping        # Update mapping

        with open(file_results, 'wb') as file:
            pickle.dump(tuple(results_list), file)  # Save updated results
        logger.info("Updated time values in the pickle file.")

    # --- Remove countries/cities with insufficient crossing detections ---
    if common.get_configs("min_crossing_detect") != 0:
        # Group values by country
        threshold: float = float(common.get_configs("min_crossing_detect"))
        country_detect: Dict[str, Dict[str, float]] = {}
        for key, value in pedestrian_cross_country.items():
            country, cond = key.rsplit('_', 1)
            val_f = float(value)
            if country not in country_detect:
                country_detect[country] = {}
            country_detect[country][cond] = val_f

        # Find countries where BOTH conditions are below threshold
        keep_countries: Set[str] = {
            country for country, vals in country_detect.items()
            if (('0' in vals or '1' in vals) and
                (vals.get('0', 0.0) + vals.get('1', 0.0) >= threshold))
        }

        df_mapping = df_mapping[df_mapping['country'].isin(keep_countries)].copy()
        # # Remove all entries in avg_speed_country and avg_time_country for those countries
        # for dict_name, d in [('avg_speed_country', avg_speed_country), ('avg_time_country', avg_time_country)]:
        #     keys_to_remove = [key for key in d if key.split('_')[0] in remove_countries]  # type: ignore
        #     for key in keys_to_remove:
        #         logger.debug(f"Deleting from {dict_name}: {key} -> {d[key]}")  # type: ignore
        #         del d[key]  # type: ignore

        # --- Remove low-detection cities from city-level speed/time ---
        # Sum all conditions for each city in pedestrian_cross_city
        # city_sum = defaultdict(int)
        # for key, value in pedestrian_cross_city.items():
        #     city = key.split('_')[0]
        #     city_sum[city] += value

        # Find cities with total crossings below threshold
        # remove_cities = {city for city, total in city_sum.items()
        #                  if total < common.get_configs("min_crossing_detect")}

        # # Remove rows from df_mapping where 'cities' is in remove_cities
        # df_mapping = df_mapping[~df_mapping['city'].isin(remove_cities)].copy()

        # # Remove all entries in avg_speed_city and avg_time_city for those cities
        # for dict_name, d in [('avg_speed_city', avg_speed_city), ('avg_time_city', avg_time_city)]:
        #     keys_to_remove = [key for key in d if key.split('_')[0] in remove_cities]  # type: ignore
        #     for key in keys_to_remove:
        #         logger.debug(f"Deleting from {dict_name}: {key} -> {d[key]}")  # type: ignore
        #         del d[key]  # type: ignore

    # Sort by continent and city, both in ascending order
    df_mapping = df_mapping.sort_values(by=["continent", "city"], ascending=[True, True])

    # Save updated mapping file in output
    os.makedirs(common.output_dir, exist_ok=True)  # check if folder
    df_mapping.to_csv(os.path.join(common.output_dir, "mapping_updated.csv"))

    logger.info("Detected:")
    logger.info(f"person: {person_counter}; bicycle: {bicycle_counter}; car: {car_counter}")
    logger.info(f"motorcycle: {motorcycle_counter}; bus: {bus_counter}; truck: {truck_counter}")
    logger.info(f"cellphone: {cellphone_counter}; traffic light: {traffic_light_counter}; " +
                f"traffic sign: {stop_sign_counter}")

    logger.info("Producing output.")

    # Data to avoid showing on hover in scatter plots
    columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type',
                      'channel']
    hover_data = list(set(df_mapping.columns) - set(columns_remove))

    df = df_mapping.copy()  # copy df to manipulate for output
    df['state'] = df['state'].fillna('NA')  # Set state to NA

    # Maps with filtered data
    plots_class.mapbox_map(df=df, hover_data=hover_data, file_name='mapbox_map')

    plots_class.mapbox_map(df=df,
                           hover_data=hover_data,
                           density_col='total_time',
                           density_radius=10,
                           file_name='mapbox_map_time')

    plots_class.world_map(df_mapping=df)  # map with countries

    plots_class.violin_plot(data_index=22, name="speed", min_threshold=common.get_configs("min_speed_limit"),
                            max_threshold=common.get_configs("max_speed_limit"), df_mapping=df_mapping, save_file=True)

    # ------------All values----------------- #
    plots_class.hist(data_index=22,
                     name="speed",
                     marginal="violin",
                     nbins=100,
                     raw=True,
                     min_threshold=common.get_configs("min_speed_limit"),
                     max_threshold=common.get_configs("max_speed_limit"),
                     font_size=common.get_configs("font_size") + 4,
                     fig_save_height=650,
                     save_file=True)

    plots_class.hist(data_index=39,
                     name="time",
                     marginal="violin",
                     # nbins=100,
                     raw=True,
                     min_threshold=None,
                     max_threshold=None,
                     font_size=common.get_configs("font_size") + 4,
                     fig_save_height=650,
                     save_file=True)

    # ------------Filtered values----------------- #
    plots_class.hist(data_index=38,
                     name="speed_filtered",
                     marginal="violin",
                     nbins=100,
                     raw=False,
                     min_threshold=common.get_configs("min_speed_limit"),
                     max_threshold=common.get_configs("max_speed_limit"),
                     font_size=common.get_configs("font_size") + 4,
                     fig_save_height=650,
                     save_file=True)

    plots_class.hist(data_index=37,
                     name="time_filtered",
                     marginal="violin",
                     # nbins=100,
                     raw=False,
                     min_threshold=None,
                     max_threshold=None,
                     font_size=common.get_configs("font_size") + 4,
                     df_mapping=df_mapping,
                     fig_save_height=650,
                     save_file=True)

    if common.get_configs("analysis_level") == "city":

        # Amount of footage
        plots_class.scatter(df=df,
                            x="total_time",
                            y="person",
                            color="continent",
                            text="city",
                            xaxis_title='Total time of footage (s)',
                            yaxis_title='Number of detected pedestrians',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.01,
                            legend_y=1.0,
                            label_distance_factor=5.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # todo: ISO-3 codes next to figures shift. need to correct once "final" dataset is online
        plots_class.speed_and_time_to_start_cross(df_mapping,
                                                  x_axis_title_height=110,
                                                  font_size_captions=common.get_configs("font_size") + 8,
                                                  legend_x=0.9,
                                                  legend_y=0.01,
                                                  legend_spacing=0.0026)

        plots_class.stack_plot(df,
                               order_by="alphabetical",
                               metric="time",
                               data_view="combined",
                               title_text="Crossing initiation time (s)",
                               filename="time_crossing_alphabetical",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=80,
                               right_margin=80
                               )

        plots_class.stack_plot(df,
                               order_by="alphabetical",
                               metric="time",
                               data_view="day",
                               title_text="Crossing initiation time (s)",
                               filename="time_crossing_alphabetical_day",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=80,
                               right_margin=80
                               )

        plots_class.stack_plot(df,
                               order_by="alphabetical",
                               metric="time",
                               data_view="night",
                               title_text="Crossing initiation time (s)",
                               filename="time_crossing_alphabetical_night",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=80,
                               right_margin=80
                               )

        plots_class.stack_plot(df,
                               order_by="average",
                               metric="time",
                               data_view="combined",
                               title_text="Crossing initiation time (s)",
                               filename="time_crossing_avg",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=10,
                               right_margin=10
                               )

        plots_class.stack_plot(df,
                               order_by="average",
                               metric="time",
                               data_view="day",
                               title_text="Crossing initiation time (s)",
                               filename="time_crossing_avg_day",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=10,
                               right_margin=10
                               )

        plots_class.stack_plot(df,
                               order_by="average",
                               metric="time",
                               data_view="night",
                               title_text="Crossing initiation time (s)",
                               filename="time_crossing_avg_night",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=10,
                               right_margin=10
                               )

        plots_class.stack_plot(df,
                               order_by="alphabetical",
                               metric="speed",
                               data_view="combined",
                               title_text="Mean speed of crossing (in m/s)",
                               filename="speed_crossing_alphabetical",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=80,
                               right_margin=80
                               )

        plots_class.stack_plot(df,
                               order_by="alphabetical",
                               metric="speed",
                               data_view="day",
                               title_text="Mean speed of crossing (in m/s)",
                               filename="speed_crossing_alphabetical_day",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=80,
                               right_margin=80
                               )

        plots_class.stack_plot(df,
                               order_by="alphabetical",
                               metric="speed",
                               data_view="night",
                               title_text="Mean speed of crossing (in m/s)",
                               filename="speed_crossing_alphabetical_night",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=80,
                               right_margin=80
                               )

        plots_class.stack_plot(df,
                               order_by="average",
                               metric="speed",
                               data_view="combined",
                               title_text="Mean speed of crossing (in m/s)",
                               filename="speed_crossing_avg",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=10,
                               right_margin=10
                               )

        plots_class.stack_plot(df,
                               order_by="average",
                               metric="speed",
                               data_view="day",
                               title_text="Mean speed of crossing (in m/s)",
                               filename="speed_crossing_avg_day",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=10,
                               right_margin=10
                               )

        plots_class.stack_plot(df,
                               order_by="average",
                               metric="speed",
                               data_view="night",
                               title_text="Mean speed of crossing (in m/s)",
                               filename="speed_crossing_avg_night",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=10,
                               right_margin=10
                               )

        plots_class.correlation_matrix(df_mapping, pedestrian_cross_city, person_city, bicycle_city, car_city,
                                       motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city,
                                       cellphone_city, traffic_sign_city, all_speed, all_time, avg_time_city,
                                       avg_speed_city)

        # Speed of crossing vs time to start crossing
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[df["time_crossing"] != 0]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="speed_crossing",
                            y="time_crossing",
                            color="continent",
                            text="city",
                            xaxis_title='Speed of crossing (in m/s)',
                            yaxis_title='Crossing initiation time (in s)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.01,
                            legend_y=1.0,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing during daytime vs time to start crossing during daytime
        df = df_mapping[df_mapping["speed_crossing_day"] != 0].copy()
        df = df[df["time_crossing_day"] != 0]
        df['state'] = df['state'].fillna('NA')

        plots_class.scatter(df=df,
                            x="speed_crossing_day",
                            y="time_crossing_day",
                            color="continent",
                            text="city",
                            xaxis_title='Crossing speed during daytime (in m/s)',
                            yaxis_title='Crossing initiation time during daytime (in s)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.01,
                            legend_y=1.0,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing during night time vs time to start crossing during night time
        df = df_mapping[df_mapping["speed_crossing_night"] != 0].copy()
        df = df[df["time_crossing_night"] != 0]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="speed_crossing_night",
                            y="time_crossing_night",
                            color="continent",
                            text="city",
                            xaxis_title='Crossing speed during night time (in m/s)',
                            yaxis_title='Crossing initiation time during night time (in s)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.8,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["population_city"].notna()) & (df["population_city"] != 0)]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="time_crossing",
                            y="population_city",
                            color="continent",
                            text="city",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='Population of city',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=2.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[(df["population_city"].notna()) & (df["population_city"] != 0)]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="speed_crossing",
                            y="population_city",
                            color="continent",
                            text="city",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Population of city',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="time_crossing",
                            y="traffic_mortality",
                            color="continent",
                            text="city",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='National traffic mortality rate (per 100,000 of population)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="speed_crossing",
                            y="traffic_mortality",
                            color="continent",
                            text="city",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='National traffic mortality rate (per 100,000 of population)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=2.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="time_crossing",
                            y="literacy_rate",
                            color="continent",
                            text="city",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='Literacy rate',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=0.01,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="speed_crossing",
                            y="literacy_rate",
                            color="continent",
                            text="city",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Literacy rate',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=0.01,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["gini"].notna()) & (df["gini"] != 0)]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="time_crossing",
                            y="gini",
                            color="continent",
                            text="city",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='Gini coefficient',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[(df["gini"].notna()) & (df["gini"] != 0)]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="speed_crossing",
                            y="gini",
                            color="continent",
                            text="city",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Gini coefficient',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=2.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df = df[(df["traffic_index"].notna()) & (df["traffic_index"] != 0)]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="time_crossing",
                            y="traffic_index",
                            color="continent",
                            text="city",
                            # size="gmp",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='Traffic index',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=2.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[df["traffic_index"] != 0]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="speed_crossing",
                            y="traffic_index",
                            color="continent",
                            text="city",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Traffic index',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=2.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs detected mobile phones
        df = df_mapping[df_mapping["time_crossing"] != 0].copy()
        df['state'] = df['state'].fillna('NA')
        df['cellphone_normalised'] = df['cellphone'] / df['total_time']
        plots_class.scatter(df=df,
                            x="time_crossing",
                            y="cellphone_normalised",
                            color="continent",
                            text="city",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='Mobile phones detected (normalised over time)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs detected mobile phones
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df['state'] = df['state'].fillna('NA')
        df['cellphone_normalised'] = df['cellphone'] / df['total_time']
        plots_class.scatter(df=df,
                            x="speed_crossing",
                            y="cellphone_normalised",
                            color="continent",
                            text="city",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Mobile phones detected (normalised over time)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Jaywalking
        plots_class.plot_crossing_without_traffic_light(df_mapping,
                                                        x_axis_title_height=60,
                                                        font_size_captions=common.get_configs("font_size"),
                                                        legend_x=0.97,
                                                        legend_y=1.0,
                                                        legend_spacing=0.004)
        plots_class.plot_crossing_with_traffic_light(df_mapping,
                                                     x_axis_title_height=60,
                                                     font_size_captions=common.get_configs("font_size"),
                                                     legend_x=0.97,
                                                     legend_y=1.0,
                                                     legend_spacing=0.004)

        # Crossing with and without traffic lights
        df = df_mapping.copy()
        df['state'] = df['state'].fillna('NA')
        df['with_trf_light_norm'] = (df['with_trf_light_day'] + df['with_trf_light_night']) / df['total_time'] / df['population_city']  # noqa: E501
        df['without_trf_light_norm'] = (df['without_trf_light_day'] + df['without_trf_light_night']) / df['total_time'] / df['population_city']  # noqa: E501
        plots_class.scatter(df=df,
                            x="with_trf_light_norm",
                            y="without_trf_light_norm",
                            color="continent",
                            text="city",
                            xaxis_title='Crossing events with traffic lights (normalised)',
                            yaxis_title='Crossing events without traffic lights (normalised)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="city",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=3.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

    if common.get_configs("analysis_level") == "country":
        df_countries = analysis_class.aggregate_by_iso3(df_mapping)
        df_countries_raw = analysis_class.aggregate_by_iso3(df_mapping_raw)

        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type']
        hover_data = list(set(df_countries.columns) - set(columns_remove))

        columns_remove_raw = ['gini', 'traffic_mortality', 'avg_height', 'population_country', 'population_city',
                              'med_age', 'literacy_rate']
        hover_data_raw = list(set(df_countries.columns) - set(columns_remove) - set(columns_remove_raw))

        df_countries.to_csv(os.path.join(common.output_dir, "mapping_countries.csv"))

        # Map with images. currently works on a 13" MacBook air screen in chrome, as things are hardcoded...
        plots_class.map_world(df=df_countries_raw,
                              color="continent",                # same default as map_political
                              show_cities=True,
                              df_cities=df_mapping,
                              show_images=True,
                              hover_data=hover_data_raw,
                              save_file=True,
                              save_final=False,
                              file_name="raw_map")

        # Map with screenshots and countries colours by continent
        plots_class.map_world(df=df_countries,
                              color="continent",
                              show_cities=True,
                              df_cities=df_mapping,
                              show_images=True,
                              hover_data=hover_data,
                              save_file=False,
                              save_final=False,
                              file_name="map_screenshots",
                              show_colorbar=True,
                              colorbar_title="Continent",
                              colorbar_kwargs=dict(y=0.035, len=0.55, bgcolor="rgba(255,255,255,0.9)"))

        # Map with screenshots and countries colours by amount of footage
        hover_data = list(set(df_countries_raw.columns) - set(columns_remove))

        # log(1 + x) to avoid -inf for zero
        df_countries_raw["log_total_time"] = np.log1p(df_countries_raw["total_time"])

        # Produce map with all data
        df = df_mapping_raw.copy()  # copy df to manipulate for output
        df['state'] = df['state'].fillna('NA')  # Set state to NA

        # Sort by continent and city, both in ascending order
        df = df.sort_values(by=["continent", "city"], ascending=[True, True])

        plots_class.map_world(df=df_countries_raw,
                              color="log_total_time",
                              show_cities=True,
                              df_cities=df_mapping,             # fixed from df to df_mapping
                              show_images=True,
                              hover_data=hover_data,
                              show_colorbar=True,
                              colorbar_title="Footage (log)",
                              save_file=True,
                              save_final=False,
                              file_name="map_screenshots_total_time")

        df_countries_raw.drop(['speed_crossing_day_country', 'speed_crossing_night_country',
                               'speed_crossing_day_night_country_avg',
                               'time_crossing_day_country', 'time_crossing_night_country',
                               'time_crossing_day_night_country_avg'
                               ], axis=1, inplace=True)
        df_countries_raw.to_csv(os.path.join(common.output_dir, "mapping_countries_raw.csv"))

        # Amount of footage
        plots_class.scatter(df=df_countries,
                            x="total_time",
                            y="person",
                            extension=common.get_configs("analysis_level"),
                            color="continent",
                            text="iso3",
                            xaxis_title='Total time of footage (s)',
                            yaxis_title='Number of detected pedestrians',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.01,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Amount of bicycle footage normalised
        df = df_countries[df_countries["person"] != 0].copy()
        df['person_norm'] = df['person'] / df['total_time']
        plots_class.scatter(df=df,
                            x="total_time",
                            y="person_norm",
                            color="continent",
                            text="iso3",
                            xaxis_title='Total time of footage (s)',
                            yaxis_title='Number of detected pedestrians (normalised over amount of footage)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.94,
                            legend_y=1.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Amount of bicycle footage normalised
        df = df_countries[df_countries["bicycle"] != 0].copy()
        df['bicycle_norm'] = df['bicycle'] / df['total_time']
        plots_class.scatter(df=df,
                            x="total_time",
                            y="bicycle_norm",
                            color="continent",
                            text="iso3",
                            xaxis_title='Total time of footage (s)',
                            yaxis_title='Number of detected bicycle (normalised over amount of footage)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.94,
                            legend_y=1.0,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="time",
                                       data_view="combined",
                                       title_text="Crossing initiation time (s)",
                                       filename="time_crossing_avg_country",
                                       font_size_captions=common.get_configs("font_size") + 8,
                                       legend_x=0.95,
                                       legend_y=0.04,
                                       legend_spacing=0.02,
                                       top_margin=100)

        plots_class.stack_plot_country(df_countries,
                                       order_by="condition",
                                       metric="speed",
                                       data_view="combined",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_combined_country",
                                       font_size_captions=common.get_configs("font_size") + 28,
                                       legend_x=0.92,
                                       legend_y=0.04,
                                       legend_spacing=0.02,
                                       top_margin=150,
                                       height=2450,
                                       width=2480)

        plots_class.stack_plot_country(df_countries,
                                       order_by="condition",
                                       metric="time",
                                       data_view="combined",
                                       title_text="Mean crossing initiation time (in s)",
                                       filename="time_crossing_combined_country",
                                       font_size_captions=common.get_configs("font_size") + 28,
                                       legend_x=0.92,
                                       legend_y=0.04,
                                       legend_spacing=0.02,
                                       top_margin=150,
                                       height=2400,
                                       width=2480)

        plots_class.stack_plot_country(df_countries_raw,
                                       order_by="condition",
                                       metric="speed",
                                       data_view="combined",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_combined_country_raw",
                                       font_size_captions=common.get_configs("font_size") + 28,
                                       raw=True,
                                       legend_x=0.92,
                                       legend_y=0.04,
                                       legend_spacing=0.02,
                                       top_margin=150,
                                       height=2400,
                                       width=2480)

        plots_class.stack_plot_country(df_countries_raw,
                                       order_by="condition",
                                       metric="time",
                                       data_view="combined",
                                       title_text="Crossing initiation time (s)",
                                       filename="time_crossing_combined_country_raw",
                                       font_size_captions=common.get_configs("font_size") + 28,
                                       raw=True,
                                       legend_x=0.92,
                                       legend_y=0.04,
                                       legend_spacing=0.02,
                                       top_margin=150,
                                       height=2400,
                                       width=2480)

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="time",
                                       data_view="combined",
                                       title_text="Crossing initiation time (s)",
                                       filename="time_crossing_alphabetical_country",
                                       font_size_captions=common.get_configs("font_size"),
                                       legend_x=0.94,
                                       legend_y=0.03,
                                       legend_spacing=0.02,
                                       top_margin=100)

        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="speed",
                                       data_view="combined",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_avg_country",
                                       font_size_captions=common.get_configs("font_size") + 8,
                                       legend_x=0.87,
                                       legend_y=0.04,
                                       legend_spacing=0.02,
                                       top_margin=100)

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="speed",
                                       data_view="combined",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_alphabetical_country",
                                       font_size_captions=common.get_configs("font_size"),
                                       legend_x=0.94,
                                       legend_y=0.03,
                                       legend_spacing=0.02,
                                       top_margin=100)

        # Plotting stacked plot during day
        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="time",
                                       data_view="day",
                                       title_text="Crossing initiation time (s)",
                                       filename="time_crossing_avg_day_country",
                                       font_size_captions=common.get_configs("font_size"),
                                       top_margin=100)

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="time",
                                       data_view="day",
                                       title_text="Crossing initiation time (s)",
                                       filename="time_crossing_alphabetical_day_country",
                                       font_size_captions=common.get_configs("font_size"),
                                       top_margin=100)

        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="speed",
                                       data_view="day",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_avg_day_country",
                                       font_size_captions=common.get_configs("font_size"),
                                       top_margin=100)

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="speed",
                                       data_view="day",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_alphabetical_day_country",
                                       font_size_captions=common.get_configs("font_size"),
                                       top_margin=100)

        # Plotting stacked plot during night
        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="time",
                                       data_view="night",
                                       title_text="Crossing initiation time (s)",
                                       filename="time_crossing_avg_night_country",
                                       font_size_captions=common.get_configs("font_size"))

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="time",
                                       data_view="night",
                                       title_text="Crossing initiation time (s)",
                                       filename="time_crossing_alphabetical_night_country",
                                       font_size_captions=common.get_configs("font_size"),
                                       top_margin=100)

        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="speed",
                                       data_view="night",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_avg_night_country",
                                       font_size_captions=common.get_configs("font_size"),
                                       top_margin=100)

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="speed",
                                       data_view="night",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_alphabetical_night_country",
                                       font_size_captions=common.get_configs("font_size"),
                                       top_margin=100)

        plots_class.speed_and_time_to_start_cross_country(df_countries,
                                                          x_axis_title_height=110,
                                                          font_size_captions=common.get_configs("font_size") + 8,
                                                          legend_x=0.87,
                                                          legend_y=0.04,
                                                          legend_spacing=0.01)

        plots_class.correlation_matrix_country(df_mapping, df_countries, pedestrian_cross_city, person_city,
                                               bicycle_city, car_city, motorcycle_city, bus_city, truck_city,
                                               cross_evnt_city, vehicle_city, cellphone_city, traffic_sign_city,
                                               avg_speed_country, avg_time_country,
                                               crossings_without_traffic_equipment_country)

        # Speed of crossing vs Crossing initiation time
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[df["time_crossing_day_night_country_avg"] != 0]
        plots_class.scatter(df=df,
                            x="speed_crossing_day_night_country_avg",
                            y="time_crossing_day_night_country_avg",
                            color="continent",
                            text="iso3",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Crossing initiation time (s)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing during daytime vs time to start crossing during daytime
        df = df_countries[df_countries["speed_crossing_day_country"] != 0].copy()
        df = df[df["time_crossing_day_country"] != 0]
        plots_class.scatter(df=df,
                            x="speed_crossing_day_country",
                            y="time_crossing_day_country",
                            color="continent",
                            text="iso3",
                            xaxis_title='Crossing speed during daytime (in m/s)',
                            yaxis_title='Crossing initiation time during daytime (in s)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing during night time vs time to start crossing during night time
        df = df_countries[df_countries["speed_crossing_night_country"] != 0].copy()
        df = df[df["time_crossing_night_country"] != 0]
        plots_class.scatter(df=df,
                            x="speed_crossing_night_country",
                            y="time_crossing_night_country",
                            color="continent",
                            text="iso3",
                            xaxis_title='Crossing speed during night time (in m/s)',
                            yaxis_title='Crossing initiation time during night time (in s)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["population_country"].notna()) & (df["population_country"] != 0)]
        plots_class.scatter(df=df,
                            x="time_crossing_day_night_country_avg",
                            y="population_country",
                            color="continent",
                            text="iso3",
                            xaxis_title='Crossing initiation time (s)',
                            yaxis_title='Population of country',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of country
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["population_country"].notna()) & (df["population_country"] != 0)]
        plots_class.scatter(df=df,
                            x="speed_crossing_day_night_country_avg",
                            y="population_country",
                            color="continent",
                            text="iso3",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Population of country',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.2,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
        plots_class.scatter(df=df,
                            x="time_crossing_day_night_country_avg",
                            y="traffic_mortality",
                            color="continent",
                            text="iso3",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='National traffic mortality rate (per 100,000 of population)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
        plots_class.scatter(df=df,
                            x="speed_crossing_day_night_country_avg",
                            y="traffic_mortality",
                            color="continent",
                            text="iso3",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='National traffic mortality rate (per 100,000 of population)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.3,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
        plots_class.scatter(df=df,
                            x="time_crossing_day_night_country_avg",
                            y="literacy_rate",
                            color="continent",
                            text="iso3",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='Literacy rate',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=0.01,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
        plots_class.scatter(df=df,
                            x="speed_crossing_day_night_country_avg",
                            y="literacy_rate",
                            color="continent",
                            text="iso3",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Literacy rate',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=0.01,
                            label_distance_factor=0.4,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["gini"].notna()) & (df["gini"] != 0)]
        plots_class.scatter(df=df,
                            x="time_crossing_day_night_country_avg",
                            y="gini",
                            color="continent",
                            text="iso3",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='Gini coefficient',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["gini"].notna()) & (df["gini"] != 0)]
        plots_class.scatter(df=df,
                            x="speed_crossing_day_night_country_avg",
                            y="gini",
                            color="continent",
                            text="iso3",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Gini coefficient',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Time to start crossing vs population of city
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df = df[(df["med_age"].notna()) & (df["med_age"] != 0)]
        plots_class.scatter(df=df,
                            x="time_crossing_day_night_country_avg",
                            y="med_age",
                            color="continent",
                            text="iso3",
                            # size="gmp",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='Median age (in years)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs population of city
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[df["med_age"] != 0]
        plots_class.scatter(df=df,
                            x="speed_crossing_day_night_country_avg",
                            y="med_age",
                            color="continent",
                            text="iso3",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Median age (in years)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.4,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs detected mobile phones
        df = df_countries[df_countries["time_crossing_day_night_country_avg"] != 0].copy()
        df['cellphone_normalised'] = df['cellphone'] / df['total_time']
        plots_class.scatter(df=df,
                            x="time_crossing_day_night_country_avg",
                            y="cellphone_normalised",
                            color="continent",
                            text="iso3",
                            xaxis_title='Crossing initiation time (in s)',
                            yaxis_title='Mobile phones detected (normalised over time)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Speed of crossing vs detected mobile phones
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df['cellphone_normalised'] = df['cellphone'] / df['total_time']
        plots_class.scatter(df=df,
                            x="speed_crossing_day_night_country_avg",
                            y="cellphone_normalised",
                            color="continent",
                            text="iso3",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Mobile phones detected (normalised over time)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Mean speed of crossing (used to be plots_class.map)
        plots_class.map_world(df=df_countries,
                              color="speed_crossing_day_night_country_avg",
                              title="Mean speed of crossing (in m/s)",
                              show_colorbar=True,
                              colorbar_title="",                 # keep your empty title behavior
                              filter_zero_nan=True,              # preserves old map() filtering
                              save_file=True,
                              file_name="map_speed_crossing"
                              )

        # Crossing initiation time (used to be plots_class.map)
        plots_class.map_world(df=df_countries,
                              color="time_crossing_day_night_country_avg",
                              title="Crossing initiation time (in s)",
                              show_colorbar=True,
                              colorbar_title="",
                              filter_zero_nan=True,
                              save_file=True,
                              file_name="map_crossing_time"
                              )

        # Crossing with and without traffic lights
        df = df_countries.copy()
        # df['state'] = df['state'].fillna('NA')
        df['with_trf_light_norm'] = (df['with_trf_light_day_country'] + df['with_trf_light_night_country']) / df['total_time'] / df['population_country']  # noqa: E501
        df['without_trf_light_norm'] = (df['without_trf_light_day_country'] + df['without_trf_light_night_country']) / df['total_time'] / df['population_country']  # noqa: E501
        df['country'] = df['country'].str.title()
        plots_class.scatter(df=df,
                            x="with_trf_light_norm",
                            y="without_trf_light_norm",
                            color="continent",
                            text="iso3",
                            xaxis_title='Crossing events with traffic lights (normalised)',
                            yaxis_title='Crossing events without traffic lights (normalised)',
                            pretty_text=False,
                            marker_size=10,
                            save_file=True,
                            hover_data=hover_data,
                            hover_name="country",
                            legend_title="",
                            legend_x=0.87,
                            legend_y=1.0,
                            label_distance_factor=0.5,
                            marginal_x=None,  # type: ignore
                            marginal_y=None)  # type: ignore

        # Exclude zero values before finding min
        nonzero_speed = df_countries[df_countries["speed_crossing_day_night_country_avg"] > 0]
        nonzero_time = df_countries[df_countries["time_crossing_day_night_country_avg"] > 0]

        max_speed_idx = df_countries["speed_crossing_day_night_country_avg"].idxmax()
        min_speed_idx = nonzero_speed["speed_crossing_day_night_country_avg"].idxmin()

        max_time_idx = df_countries["time_crossing_day_night_country_avg"].idxmax()
        min_time_idx = nonzero_time["time_crossing_day_night_country_avg"].idxmin()

        # Mean and standard deviation
        speed_mean = nonzero_speed["speed_crossing_day_night_country_avg"].mean()
        speed_std = nonzero_speed["speed_crossing_day_night_country_avg"].std()

        time_mean = nonzero_time["time_crossing_day_night_country_avg"].mean()
        time_std = nonzero_time["time_crossing_day_night_country_avg"].std()

        logger.info(f"Country with the highest average speed while crossing: {df_countries.loc[max_speed_idx, 'country']} "  # noqa:E501
                    f"({df_countries.loc[max_speed_idx, 'speed_crossing_day_night_country_avg']:.2f})")

        logger.info(f"Country with the lowest non-zero average speed while crossing: {nonzero_speed.loc[min_speed_idx, 'country']} "  # noqa:E501
                    f"({nonzero_speed.loc[min_speed_idx, 'speed_crossing_day_night_country_avg']:.2f})")

        logger.info(f"Mean speed while crossing (non-zero): {speed_mean:.2f}")
        logger.info(f"Standard deviation of speed while crossing (non-zero): {speed_std:.2f}")

        logger.info(f"Country with the highest average crossing time: {df_countries.loc[max_time_idx, 'country']} "
                    f"({df_countries.loc[max_time_idx, 'time_crossing_day_night_country_avg']:.2f})")

        logger.info(f"Country with the lowest non-zero average crossing time: {nonzero_time.loc[min_time_idx, 'country']} "  # noqa: E501
                    f"({nonzero_time.loc[min_time_idx, 'time_crossing_day_night_country_avg']:.2f})")

        logger.info(f"Mean crossing time (non-zero): {time_mean:.2f}")
        logger.info(f"Standard deviation of crossing time (non-zero): {time_std:.2f}")

        stats = df_countries[['total_time', 'total_videos']].agg(['mean', 'std', 'sum'])

        logger.info(
            f"Average total_time: {stats.loc['mean', 'total_time']:.2f}, "
            f"Standard deviation: {stats.loc['std', 'total_time']:.2f}, "
            f"Sum: {stats.loc['sum', 'total_time']:.2f}"
        )
        logger.info(
            f"Average total_videos: {stats.loc['mean', 'total_videos']:.2f}, "
            f"Standard deviation: {stats.loc['std', 'total_videos']:.2f}, "
            f"Sum: {stats.loc['sum', 'total_videos']:.2f}"
        )

        # Max total_time
        max_row = df_countries.loc[df_countries['total_time'].idxmax()]
        logger.info(
            f"Country with maximum total_time: {max_row['country']}, "
            f"total_time: {max_row['total_time']}, "
            f"total_videos: {max_row['total_videos']}"
        )

        # Min total_time
        min_row = df_countries.loc[df_countries['total_time'].idxmin()]
        logger.info(
            f"Country with minimum total_time: {min_row['country']}, "
            f"total_time: {min_row['total_time']}, "
            f"total_videos: {min_row['total_videos']}"
        )
