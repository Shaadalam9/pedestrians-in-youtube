# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import math
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import heapq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helper_script import Youtube_Helper
from utils.algorithms import Algorithms
from utils.values import Values
from utils.wrappers import Wrappers
from utils.plot import Plots
from utils.geometry import Geometry
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
FLAG_SIZE = 12
TEXT_SIZE = 12
SCALE = 1  # scale=3 hangs often

video_paths = common.get_configs("videos")


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
        file = self.clean_csv_filename(file)
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

            # Check if the footage duration meets the minimum threshold
            total_seconds = values_class.calculate_total_seconds_for_city(
                df_mapping, values[4], values[5]
            )

            if total_seconds <= common.get_configs("footage_threshold"):
                return None  # Skip if not enough seconds

            file_path = os.path.join(folder_path, file)
            try:
                logger.debug(f"Adding file {file_path} to dfs.")

                # Read the CSV into a DataFrame
                df = pd.read_csv(file_path)

                # Skip if "Frame Count" column is not present
                if "Frame Count" not in df.columns:
                    logger.debug(f"Skipping non-numeric feature: {filename}")
                    return None

                # Optionally apply geometry correction if configured and not zero
                use_geom_correction = common.get_configs("use_geometry_correction")
                if use_geom_correction != 0:
                    df = geometry_class.reassign_ids_directional_cross_fix(
                        df,
                        distance_threshold=use_geom_correction,
                        yolo_ids=[0]
                    )
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}.")
                return  # Skip to the next file if reading fails
        return file

    @staticmethod
    def count_object(dataframe, id):
        """
        Counts the number of unique instances of an object with a specific ID in a DataFrame.

        Args:
            dataframe (DataFrame): The DataFrame containing object data.
            id (int): The unique ID assigned to the object.

        Returns:
            int: The number of unique instances of the object with the specified ID.
        """

        # Filter the DataFrame to include only entries for the specified object ID
        crossed_ids = dataframe[(dataframe["YOLO_id"] == id)]

        # Group the filtered data by Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

        # Count the number of groups, which represents the number of unique instances of the object
        num_groups = crossed_ids_grouped.ngroups

        return num_groups

    @staticmethod
    def calculate_total_seconds(df):
        """Calculates the total seconds of the total video according to mapping file."""
        grand_total_seconds = 0

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Extracting data from the DataFrame row

            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])

            # Iterate through each start time and end time
            for start, end in zip(start_times, end_times):
                for s, e in zip(start, end):
                    grand_total_seconds += (int(e) - int(s))

        return grand_total_seconds

    @staticmethod
    def calculate_total_videos(df):
        """
        Calculates the total number of videos in the mapping file.
        """
        total_videos = set()
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            videos = row["videos"]

            videos_list = videos.split(",")  # Split by comma to convert string to list

            for video in videos_list:
                total_videos.add(video.strip())  # Add the video to the set (removing any extra whitespace)

        return len(total_videos)

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
    _all_metrics_cache = None  # class-level cache for all metrics

    @classmethod
    def _compute_all_metrics(cls, df_mapping):
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
            - "cell_phones": cell phone detections per person, normalized for time
            - "traffic_signs": count of detected traffic signs (YOLO ids 9, 11)
            - "vehicles": count of all vehicles (YOLO ids 2, 3, 5, 7)
            - "bicycles": count of bicycles (YOLO id 1)
            - "cars": count of cars (YOLO id 3)
            - "motorcycles": count of motorcycles (YOLO id 2)
            - "buses": count of buses (YOLO id 5)
            - "trucks": count of trucks (YOLO id 7)
            - "persons": count of people (YOLO id 0)

        Note:
            - The CSV files must follow a naming pattern <video_id>_<start_time>.csv and reside in data folders
                as configured via `common.get_configs('data')`.
            - Each CSV must contain at least the columns "YOLO_id" and "Unique Id".
            - Progress is tracked using tqdm.
        """

        # List of data folders containing detection CSVs
        data_folders = common.get_configs('data')
        csv_files = {}

        # Index all CSV files from all configured folders for quick lookup
        for folder_path in data_folders:
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    csv_files[file] = os.path.join(folder_path, file)

        # Prepare result containers for each metric type
        cell_phone_info = {}
        traffic_signs_layer = {}
        vehicle_layer = {}
        bicycle_layer = {}
        car_layer = {}
        motorcycle_layer = {}
        bus_layer = {}
        truck_layer = {}
        person_layer = {}

        # Process each mapping row (one or more videos per row)
        for _, row in tqdm(df_mapping.iterrows(), total=df_mapping.shape[0], desc="Analysing the csv files:"):
            video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
            start_times = ast.literal_eval(row["start_time"])
            time_of_day = ast.literal_eval(row["time_of_day"])

            # Loop through all video_id + start_time pairs
            for vid, start_times_list, time_of_day_list in zip(video_ids, start_times, time_of_day):
                for start_time, time_of_day_value in zip(start_times_list, time_of_day_list):
                    filename = f"{vid}_{start_time}.csv"
                    if filename not in csv_files:
                        continue  # No detection CSV for this video segment
                    file_path = csv_files[filename]

                    # Find video meta details (start, end, city, location, etc.)
                    result = values_class.find_values_with_video_id(df_mapping, f"{vid}_{start_time}")
                    if result is None:
                        continue

                    start = result[1]
                    end = result[2]
                    condition = result[3]
                    city = result[4]
                    lat = result[6]
                    long = result[7]
                    duration = end - start  # Duration in seconds
                    city_id_format = f'{city}_{lat}_{long}_{condition}'  # noqa:F841
                    video_key = f"{vid}_{start_time}"

                    # Load detection data for this video segment
                    dataframe = pd.read_csv(file_path)

                    # ---- CELL PHONES: Count per person, normalised ----
                    mobile_ids = len(dataframe[dataframe["YOLO_id"] == 67]["Unique Id"].unique())
                    num_person = len(dataframe[dataframe["YOLO_id"] == 0]["Unique Id"].unique())
                    if num_person > 0 and mobile_ids > 0:
                        avg_cell_phone = ((mobile_ids * 60) / duration / num_person) * 1000
                        cell_phone_info[video_key] = avg_cell_phone

                    # ---- TRAFFIC SIGNS (YOLO 9, 11) ----
                    traffic_sign_ids = dataframe[dataframe["YOLO_id"].isin([9, 11])]["Unique Id"].unique()
                    count = (len(traffic_sign_ids) / duration) * 60 if duration > 0 else 0
                    traffic_signs_layer[video_key] = count

                    # ---- VEHICLES (YOLO 2,3,5,7) ----
                    vehicles_mask = dataframe["YOLO_id"].isin([2, 3, 5, 7])
                    vehicle_ids = dataframe[vehicles_mask]["Unique Id"].unique()
                    count = (len(vehicle_ids) / duration) * 60 if duration > 0 else 0
                    vehicle_layer[video_key] = count

                    # ---- BICYCLES (YOLO 1) ----
                    bicycle_ids = dataframe[dataframe["YOLO_id"] == 1]["Unique Id"].unique()
                    count = (len(bicycle_ids) / duration) * 60 if duration > 0 else 0
                    bicycle_layer[video_key] = count

                    # ---- CARS (YOLO 2) ----
                    car_ids = dataframe[dataframe["YOLO_id"] == 2]["Unique Id"].unique()
                    count = (len(car_ids) / duration) * 60 if duration > 0 else 0
                    car_layer[video_key] = count

                    # ---- MOTORCYCLES (YOLO 3) ----
                    motorcycle_ids = dataframe[dataframe["YOLO_id"] == 3]["Unique Id"].unique()
                    count = (len(motorcycle_ids) / duration) * 60 if duration > 0 else 0
                    motorcycle_layer[video_key] = count

                    # ---- BUSES (YOLO 5) ----
                    bus_ids = dataframe[dataframe["YOLO_id"] == 5]["Unique Id"].unique()
                    count = (len(bus_ids) / duration) * 60 if duration > 0 else 0
                    bus_layer[video_key] = count

                    # ---- TRUCKS (YOLO 7) ----
                    truck_ids = dataframe[dataframe["YOLO_id"] == 7]["Unique Id"].unique()
                    count = (len(truck_ids) / duration) * 60 if duration > 0 else 0
                    truck_layer[video_key] = count

                    # ---- PERSONS (YOLO 0) ----
                    person_ids = dataframe[dataframe["YOLO_id"] == 0]["Unique Id"].unique()
                    count = (len(person_ids) / duration) * 60 if duration > 0 else 0
                    person_layer[video_key] = count

        # --- WRAPPING AS CITY_LONGITUDE_LATITUDE_CONDITION ---
        metric_dicts = [
            ("cell_phones", cell_phone_info),
            ("traffic_signs", traffic_signs_layer),
            ("vehicles", vehicle_layer),
            ("bicycles", bicycle_layer),
            ("cars", car_layer),
            ("motorcycles", motorcycle_layer),
            ("buses", bus_layer),
            ("trucks", truck_layer),
            ("persons", person_layer),
        ]

        cls._all_metrics_cache = {}

        for i, (metric_name, metric_layer) in enumerate(metric_dicts, 1):
            logger.info(f"[{i}/{len(metric_dicts)}] Wrapping '{metric_name}' ...")
            wrapped = wrapper_class.city_country_wrapper(
                input_dict=metric_layer,
                mapping=df_mapping,
                show_progress=True  # or False if you don't want tqdm inside
            )
            cls._all_metrics_cache[metric_name] = wrapped

    @classmethod
    def _ensure_cache(cls, df_mapping):
        """
        Ensure that the class-level metrics cache is populated.
        If the cache is empty, computes all metrics for the provided mapping DataFrame.
        """
        if cls._all_metrics_cache is None:
            cls._compute_all_metrics(df_mapping)

    @classmethod
    def calculate_cell_phones(cls, df_mapping):
        """
        Return the cached cell phone metric, computing all metrics if needed.
        Raises:
            RuntimeError: If cache population fails.
        """
        cls._ensure_cache(df_mapping)
        if cls._all_metrics_cache is None:
            raise RuntimeError("Metric cache not populated.")
        return cls._all_metrics_cache["cell_phones"]

    @classmethod
    def calculate_traffic_signs(cls, df_mapping):
        """
        Return the cached traffic sign metric, computing all metrics if needed.
        Raises:
            RuntimeError: If cache population fails.
        """
        cls._ensure_cache(df_mapping)
        if cls._all_metrics_cache is None:
            raise RuntimeError("Metric cache not populated after ensure_cache!")
        return cls._all_metrics_cache["traffic_signs"]

    @classmethod
    def calculate_traffic(cls, df_mapping, person=0, bicycle=0, motorcycle=0, car=0, bus=0, truck=0):
        """
        Return the requested vehicle/person/bicycle metric from the cache, computing if needed.
        Arguments specify which traffic metric to return. If multiple flags are set, precedence is given as:
        - 'person' if set
        - 'bicycle' if set
        - if all of motorcycle, car, bus, truck are set: returns 'vehicles'
        - otherwise, returns individual type if its flag is set
        - fallback is 'vehicles'
        Raises:
            RuntimeError: If cache population fails.
        """
        cls._ensure_cache(df_mapping)
        if cls._all_metrics_cache is None:
            raise RuntimeError("Metric cache not populated after ensure_cache!")
        # Return the right metric by flags
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

    @staticmethod
    def crossing_event_with_traffic_equipment(df_mapping, data):
        """
        Analyze pedestrian crossing events in relation to the presence of traffic equipment (YOLO_id 9 or 11).

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
                    for file in os.listdir(folder_path):
                        if os.path.splitext(file)[0] == video_key:
                            file_path = os.path.join(folder_path, file)
                            value = pd.read_csv(file_path)
                            break
                    if value is not None:
                        break

                if value is None:
                    continue  # Skip if file not found

                # Analyze crossings for presence of traffic equipment
                for unique_id, _ in crossings.items():
                    unique_id_indices = value.index[value['Unique Id'] == unique_id]
                    if unique_id_indices.empty:
                        continue  # Skip if no occurrences

                    first_occurrence = unique_id_indices[0]
                    last_occurrence = unique_id_indices[-1]

                    yolo_ids = value.loc[first_occurrence:last_occurrence, 'YOLO_id']

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
                    for file in os.listdir(folder_path):
                        filename_no_ext = os.path.splitext(file)[0]
                        if filename_no_ext == key:
                            file_path = os.path.join(folder_path, file)
                            # Load the CSV
                            value = pd.read_csv(file_path)

                for id, time in df.items():
                    unique_id_indices = value.index[value['Unique Id'] == id]
                    first_occurrence = unique_id_indices[0]
                    last_occurrence = unique_id_indices[-1]

                    # Check if YOLO_id = 9 exists within the specified index range
                    yolo_id_9_exists = any(
                        value.loc[first_occurrence:last_occurrence, 'YOLO_id'] == 9)
                    yolo_id_9_not_exists = not any(
                        value.loc[first_occurrence:last_occurrence, 'YOLO_id'] == 9)

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
    def pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping):
        final = {}
        count = {key: len(value['ids']) for key, value in pedestrian_crossing_count.items()}

        for key, df in count.items():
            result = values_class.find_values_with_video_id(df_mapping, key)

            if result is not None:
                condition = result[3]
                city = result[4]
                lat = result[6]
                long = result[7]

                # Create the city_time_key (city + time_of_day)
                city_time_key = f'{city}_{lat}_{long}_{condition}'

                # Add the count to the corresponding city_time_key in the final dict
                if city_time_key in final:
                    final[city_time_key] += count[key]  # Add the current count to the existing sum
                else:
                    final[city_time_key] = count[key]

        return final

    @staticmethod
    def pedestrian_cross_per_country(pedestrian_cross_city, df_mapping):
        final = {}
        for city_lat_long_cond, value in pedestrian_cross_city.items():
            city, lat, _, cond = city_lat_long_cond.split("_")
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            if country in final:
                final[country] += value
            else:
                final[country] = value

        return final

    @staticmethod
    def safe_average(values):
        # Filter out None and NaN values.
        valid_values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(valid_values) / len(valid_values) if valid_values else 0

    @staticmethod
    def plot_crossing_without_traffic_light(df_mapping, font_size_captions=40, x_axis_title_height=150, legend_x=0.92,
                                            legend_y=0.015, legend_spacing=0.02):
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        without_trf_light = data_tuple[28]
        # Now populate the final_dict with city-wise speed data
        for city_condition, count in without_trf_light.items():
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:
                # Initialise the city's dictionary if not already present
                if f"{city}_{lat}_{long}" not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {"without_trf_light_0": None, "without_trf_light_1": None,
                                                          "country": country, "iso": iso_code}

                # normalise by total time and total number of detected persons
                total_time = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "total_time")
                person = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "person")
                count = count / total_time / person

                # Populate the corresponding speed based on the condition
                final_dict[f"{city}_{lat}_{long}"][f"without_trf_light_{condition}"] = count

        # Multiply each of the numeric speed values by 10^6
        for city_key, data in final_dict.items():
            for key, value in data.items():
                # Only modify keys that represent speed values
                if key.startswith("without_trf_light") and value is not None:
                    data[key] = round(value * 10**6, 2)

        cities_ordered = sorted(
            final_dict.keys(),
            key=lambda city: Analysis.safe_average([
                final_dict[city]["without_trf_light_0"],
                final_dict[city]["without_trf_light_1"]
            ]),
            reverse=True
        )

        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        # Prepare data for day and night stacking
        day_crossing = [final_dict[city]['without_trf_light_0'] for city in cities_ordered]
        night_crossing = [final_dict[city]['without_trf_light_1'] for city in cities_ordered]

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups
        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            if day_crossing[i] is not None and night_crossing[i] is not None:
                value = round((day_crossing[i] + night_crossing[i])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=1)

            elif day_crossing[i] is not None:  # Only day data available
                value = (day_crossing[i])
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_crossing[i] is not None:  # Only night data available
                value = (night_crossing[i])
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            idx = num_cities_per_col + i
            if day_crossing[idx] is not None and night_crossing[idx] is not None:
                value = round((day_crossing[idx] + night_crossing[idx])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_crossing[idx] is not None:
                value = (day_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_crossing[idx] is not None:
                value = (night_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_crossing[i] if day_crossing[i] is not None else 0) +
            (night_crossing[i] if night_crossing[i] is not None else 0)
            for i in range(len(cities))
        ]) if cities else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=True
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=True
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(title=dict(text="Road crossings without traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=1)

        fig.update_xaxes(title=dict(text="Road crossings without traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=2)

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=TALL_FIG_HEIGHT, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Manually add gridlines using `shapes`
        x_grid_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600]  # Define the gridline positions on the x-axis

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x', yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x2', yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Define the legend items
        legend_items = [
            {"name": "Day", "color": bar_colour_1},
            {"name": "Night", "color": bar_colour_2},
        ]

        # Add the vertical legends at the top and bottom
        plots_class.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
                                                    spacing=legend_spacing, font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=1, x1=0.495, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.505, y0=1, x1=1, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Split cities into left and right columns
        left_column_cities = cities_ordered[:num_cities_per_col]
        right_column_cities = cities_ordered[num_cities_per_col:]

        # Initialise variables for dynamic y positioning for both columns
        current_row_left = 1  # Start from the first row for the left column
        current_row_right = 1  # Start from the first row for the right column
        y_position_map_left = {}  # Store y positions for each country (left column)
        y_position_map_right = {}  # Store y positions for each country (right column)

        # Calculate the y positions dynamically for the left column
        for city in left_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_left:  # Add the country label once per country
                y_position_map_left[country] = 1 - (current_row_left - 1) / (len(left_column_cities) * 2)

            current_row_left += 2  # Increment the row for each city (speed and time take two rows)

        # Calculate the y positions dynamically for the right column
        for city in right_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_right:  # Add the country label once per country
                y_position_map_right[country] = 1 - (current_row_right - 1) / (len(right_column_cities) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        fig.update_yaxes(
            tickfont=dict(size=12, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=x_axis_title_height, b=180))
        plots_class.save_plotly_figure(fig,
                                       "crossings_without_traffic_equipment_avg",
                                       width=2480,
                                       height=TALL_FIG_HEIGHT,
                                       scale=SCALE,
                                       save_eps=False,
                                       save_final=True)

    @staticmethod
    def plot_crossing_with_traffic_light(df_mapping, font_size_captions=40, x_axis_title_height=150, legend_x=0.92,
                                         legend_y=0.015, legend_spacing=0.02):
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        with_trf_light = data_tuple[27]
        # Now populate the final_dict with city-wise speed data
        for city_condition, count in with_trf_light.items():
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:
                # Initialise the city's dictionary if not already present
                if f"{city}_{lat}_{long}" not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {"with_trf_light_0": None, "with_trf_light_1": None,
                                                          "country": country, "iso": iso_code}

                # normalise by total time and total number of detected persons
                total_time = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "total_time")
                person = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "person")
                count = count / total_time / person

                # Populate the corresponding speed based on the condition
                final_dict[f"{city}_{lat}_{long}"][f"with_trf_light_{condition}"] = count

        # Multiply each of the numeric speed values by 10^6
        for city_key, data in final_dict.items():
            for key, value in data.items():
                # Only modify keys that represent speed values
                if key.startswith("with_trf_light") and value is not None:
                    data[key] = round(value * 10**6, 2)

        cities_ordered = sorted(
            final_dict.keys(),
            key=lambda city: Analysis.safe_average([
                final_dict[city]["with_trf_light_0"],
                final_dict[city]["with_trf_light_1"]
            ]),
            reverse=True
        )

        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        # Prepare data for day and night stacking
        day_crossing = [final_dict[city]['with_trf_light_0'] for city in cities_ordered]
        night_crossing = [final_dict[city]['with_trf_light_1'] for city in cities_ordered]

        # # Ensure that plotting uses cities_ordered
        # assert len(cities_ordered) == len(day_crossing) == len(night_crossing), "Lengths of lists don't match!"

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups
        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            if day_crossing[i] is not None and night_crossing[i] is not None:
                value = round((day_crossing[i] + night_crossing[i])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_crossing[i] is not None:  # Only day data available
                value = (day_crossing[i])
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_crossing[i] is not None:  # Only night data available
                value = (night_crossing[i])
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            idx = num_cities_per_col + i
            if day_crossing[idx] is not None and night_crossing[idx] is not None:
                value = round((day_crossing[idx] + night_crossing[idx])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_crossing[idx] is not None:
                value = (day_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_crossing[idx] is not None:
                value = (night_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_crossing[i] if day_crossing[i] is not None else 0) +
            (night_crossing[i] if night_crossing[i] is not None else 0)
            for i in range(len(cities))
        ]) if cities else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=True
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=True
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(title=dict(text="Road crossings with traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=1)

        fig.update_xaxes(title=dict(text="Road crossings with traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=2)

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=TALL_FIG_HEIGHT, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Manually add gridlines using `shapes`
        x_grid_values = [50, 100, 150, 200, 250]  # Define the gridline positions on the x-axis

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x', yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x2', yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Define the legend items
        legend_items = [
            {"name": "Day", "color": bar_colour_1},
            {"name": "Night", "color": bar_colour_2},
        ]

        # Add the vertical legends at the top and bottom
        plots_class.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
                                                    spacing=legend_spacing, font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=1, x1=0.495, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.505, y0=1, x1=1, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Split cities into left and right columns
        left_column_cities = cities_ordered[:num_cities_per_col]
        right_column_cities = cities_ordered[num_cities_per_col:]

        # Initialise variables for dynamic y positioning for both columns
        current_row_left = 1  # Start from the first row for the left column
        current_row_right = 1  # Start from the first row for the right column
        y_position_map_left = {}  # Store y positions for each country (left column)
        y_position_map_right = {}  # Store y positions for each country (right column)

        # Calculate the y positions dynamically for the left column
        for city in left_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_left:  # Add the country label once per country
                y_position_map_left[country] = 1 - (current_row_left - 1) / (len(left_column_cities) * 2)

            current_row_left += 2  # Increment the row for each city (speed and time take two rows)

        # Calculate the y positions dynamically for the right column
        for city in right_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_right:  # Add the country label once per country
                y_position_map_right[country] = 1 - (current_row_right - 1) / (len(right_column_cities) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        fig.update_yaxes(
            tickfont=dict(size=12, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=x_axis_title_height, b=180))
        plots_class.save_plotly_figure(fig,
                                       "crossings_with_traffic_equipment_avg",
                                       width=2480,
                                       height=TALL_FIG_HEIGHT,
                                       scale=SCALE,
                                       save_eps=False,
                                       save_final=True)

    def compute_avg_variable_city(self, variable_city):
        """
        Compute the average value for each city-condition key in a nested dictionary.
        """
        avg_dict = {}

        for key, inner_dict in variable_city.items():
            # Compute average
            values = list(inner_dict.values())
            avg_value = sum(values) / len(values) if values else 0

            # Assign average directly to the same key
            avg_dict[key] = avg_value

        return avg_dict

    @staticmethod
    def correlation_matrix(df_mapping):
        """
        Compute and visualise correlation matrices for various city-level traffic and demographic data.

        This method:
        - Loads precomputed statistical data from a pickled file.
        - Aggregates metrics like speed, time, vehicle/pedestrian counts, and socioeconomic indicators.
        - Constructs structured dictionaries for day (condition 0) and night (condition 1) conditions.
        - Computes Spearman correlation matrices for:
            - Daytime data
            - Nighttime data
            - Averaged across both conditions
            - Per continent basis
        - Generates and saves Plotly heatmaps for all computed correlation matrices.

        Args:
            df_mapping (pd.DataFrame): A mapping DataFrame containing metadata for each city-state combination,
                                       including country, continent, GDP, literacy rate, etc.

        Raises:
            ValueError: If essential data (e.g., average speed or time) is missing.
        """
        logger.info("Plotting correlation matrices.")
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        (ped_cross_city, ped_crossing_count, person_city, bicycle_city, car_city,
         motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city,
         cellphone_city, trf_sign_city, speed_values, time_values, avg_time, avg_speed) = data_tuple[10:26]

        # Check if both 'speed' and 'time' are valid dictionaries
        if avg_speed is None or avg_time is None:
            raise ValueError("Either 'speed' or 'time' returned None, please check the input data or calculations.")

        # Remove the ones where there is data missing for a specific country and condition
        common_keys = avg_speed.keys() & avg_time.keys()

        # Retain only the key-value pairs where the key is present in both dictionaries
        avg_speed = {key: avg_speed[key] for key in common_keys}
        avg_time = {key: avg_time[key] for key in common_keys}

        # Now populate the final_dict with city-wise data
        for city_condition, speed in avg_speed.items():
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            continent = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "continent")
            population_country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "population_country")  # noqa: E501
            gdp_city = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "gmp")
            traffic_mortality = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "traffic_mortality")  # noqa: E501
            literacy_rate = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "literacy_rate")
            gini = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "gini")
            traffic_index = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "traffic_index")

            if country or iso_code is not None:

                # Initialise the city's dictionary if not already present
                if f'{city}_{lat}_{long}' not in final_dict:
                    final_dict[f'{city}_{lat}_{long}'] = {
                        "avg_speed_0": None, "avg_speed_1": None, "avg_time_0": None, "avg_time_1": None,
                        "speed_val_0": None, "speed_val_1": None, "time_val_0": None, "time_val_1": None,
                        "ped_cross_city_0": 0, "ped_cross_city_1": 0,
                        "person_city_0": 0, "person_city_1": 0, "bicycle_city_0": 0,
                        "bicycle_city_1": 0, "car_city_0": 0, "car_city_1": 0,
                        "motorcycle_city_0": 0, "motorcycle_city_1": 0, "bus_city_0": 0,
                        "bus_city_1": 0, "truck_city_0": 0, "truck_city_1": 0,
                        "cross_evnt_city_0": 0, "cross_evnt_city_1": 0, "vehicle_city_0": 0,
                        "vehicle_city_1": 0, "cellphone_city_0": 0, "cellphone_city_1": 0,
                        "trf_sign_city_0": 0, "trf_sign_city_1": 0,
                    }

                # Populate the corresponding speed and time based on the condition
                final_dict[f'{city}_{lat}_{long}'][f"avg_speed_{condition}"] = speed
                if f'{city}_{lat}_{long}_{condition}' in avg_time:
                    final_dict[f'{city}_{lat}_{long}'][f"avg_time_{condition}"] = avg_time.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"time_val_{condition}"] = time_values.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"speed_val_{condition}"] = speed_values.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"time_val_{condition}"] = time_values.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"ped_cross_city_{condition}"] = ped_cross_city.get(
                        f'{city}_{lat}_{long}_{condition}', None)

                    avg_person_city = analysis_class.compute_avg_variable_city(person_city)
                    final_dict[f'{city}_{lat}_{long}'][f"person_city_{condition}"] = avg_person_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_bicycle_city = analysis_class.compute_avg_variable_city(bicycle_city)
                    final_dict[f'{city}_{lat}_{long}'][f"bicycle_city_{condition}"] = avg_bicycle_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_car_city = analysis_class.compute_avg_variable_city(car_city)
                    final_dict[f'{city}_{lat}_{long}'][f"car_city_{condition}"] = avg_car_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_motorcycle_city = analysis_class.compute_avg_variable_city(motorcycle_city)
                    final_dict[f'{city}_{lat}_{long}'][f"motorcycle_city_{condition}"] = avg_motorcycle_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_bus_city = analysis_class.compute_avg_variable_city(bus_city)
                    final_dict[f'{city}_{lat}_{long}'][f"bus_city_{condition}"] = avg_bus_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_truck_city = analysis_class.compute_avg_variable_city(truck_city)
                    final_dict[f'{city}_{lat}_{long}'][f"truck_city_{condition}"] = avg_truck_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    final_dict[f'{city}_{lat}_{long}'][f"cross_evnt_city_{condition}"] = cross_evnt_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_vehicle_city = analysis_class.compute_avg_variable_city(vehicle_city)
                    final_dict[f'{city}_{lat}_{long}'][f"vehicle_city_{condition}"] = avg_vehicle_city.get(
                        f'{city}_{lat}_{long}_{condition}', None)

                    avg_cellphone_city = analysis_class.compute_avg_variable_city(cellphone_city)
                    final_dict[f'{city}_{lat}_{long}'][f"cellphone_city_{condition}"] = avg_cellphone_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_trf_sign_city = analysis_class.compute_avg_variable_city(trf_sign_city)
                    final_dict[f'{city}_{lat}_{long}'][f"trf_sign_city_{condition}"] = avg_trf_sign_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    final_dict[f'{city}_{lat}_{long}'][f"traffic_mortality_{condition}"] = traffic_mortality
                    final_dict[f'{city}_{lat}_{long}'][f"literacy_rate_{condition}"] = literacy_rate
                    final_dict[f'{city}_{lat}_{long}'][f"gini_{condition}"] = gini
                    final_dict[f'{city}_{lat}_{long}'][f"traffic_index_{condition}"] = traffic_index
                    final_dict[f'{city}_{lat}_{long}'][f"continent_{condition}"] = continent
                    if gdp_city is not None:
                        final_dict[f'{city}_{lat}_{long}'][f"gmp_{condition}"] = gdp_city/population_country

        # Initialise an empty list to store the rows for the DataFrame
        data_day, data_night = [], []

        # Loop over each city and gather relevant values for condition 0
        for city in final_dict:
            # Initialise a dictionary for the row
            row_day, row_night = {}, {}

            # Add data for condition 0 (ignore 'speed_val' and 'time_val')
            for condition in ['0']:  # Only include condition 0
                for key, value in final_dict[city].items():
                    if condition in key and 'speed_val' not in key and 'time_val' not in key and 'continent' not in key:  # noqa:E501
                        row_day[key] = value

            # Append the row to the data list
            data_day.append(row_day)

            for condition in ['1']:  # Only include condition 1
                for key, value in final_dict[city].items():
                    if condition in key and 'speed_val' not in key and 'time_val' not in key and 'continent' not in key:  # noqa:E501
                        row_night[key] = value

            # Append the row to the data list
            data_night.append(row_night)

        # Convert the list of rows into a Pandas DataFrame
        df_day = pd.DataFrame(data_day)
        df_night = pd.DataFrame(data_night)

        # Calculate the correlation matrix
        corr_matrix_day = df_day.corr(method='spearman')
        corr_matrix_night = df_night.corr(method='spearman')

        # Rename the variables in the correlation matrix
        rename_dict_1 = {
            'avg_speed_0': 'Speed of', 'avg_speed_1': 'Crossing speed',
            'avg_time_0': 'Crossing decision time', 'avg_time_1': 'Crossing decision time',
            'ped_cross_city_0': 'Crossing', 'ped_cross_city_1': 'Crossing',
            'person_city_0': 'Detected persons', 'person_city_1': 'Detected persons',
            'bicycle_city_0': 'Detected bicycles', 'bicycle_city_1': 'Detected bicycles',
            'car_city_0': 'Detected cars', 'car_city_1': 'Detected cars',
            'motorcycle_city_0': 'Detected motorcycles', 'motorcycle_city_1': 'Detected motorcycles',
            'bus_city_0': 'Detected bus', 'bus_city_1': 'Detected bus',
            'truck_city_0': 'Detected truck', 'truck_city_1': 'Detected truck',
            'cross_evnt_city_0': 'Detected crossings without traffic lights',
            'cross_evnt_city_1': 'Detected crossings without traffic lights',
            'vehicle_city_0': 'Detected motor vehicles',
            'vehicle_city_1': 'Detected motor vehicles',
            'cellphone_city_0': 'Detected cellphones', 'cellphone_city_1': 'Detected cellphones',
            'trf_sign_city_0': 'Detected traffic signs', 'trf_sign_city_1': 'Detected traffic signs',
            'gmp_0': 'GMP', 'gmp_1': 'GMP',
            'traffic_mortality_0': 'Traffic mortality', 'traffic_mortality_1': 'Traffic mortality',
            'literacy_rate_0': 'Literacy rate', 'literacy_rate_1': 'Literacy rate',
            'gini_0': 'Gini coefficient', 'gini_1': 'Gini coefficient', 'traffic_index_0': 'Traffic index',
            'traffic_index_1': 'Traffic index'
            }

        corr_matrix_day = corr_matrix_day.rename(columns=rename_dict_1, index=rename_dict_1)
        corr_matrix_night = corr_matrix_night.rename(columns=rename_dict_1, index=rename_dict_1)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_day, text_auto=".2f",  # Display correlation values on the heatmap # type: ignore
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto")  # Automatically adjust aspect ratio
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        plots_class.save_plotly_figure(fig, "correlation_matrix_heatmap_day", save_final=True)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_night, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation Matrix Heatmap in night"  # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        plots_class.save_plotly_figure(fig, "correlation_matrix_heatmap_night", save_final=True)

        # Initialise a list to store rows of data (one row per city)
        data_rows = []

        # Assuming `conditions` is a list of conditions you are working with
        conditions = ['0', '1']  # Modify this list to include all conditions you have (e.g., '0', '1', etc.)

        # Iterate over each city and condition
        for city in final_dict:
            # Initialise a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'gmp', 'traffic_mortality', 'literacy_rate', 'gini', 'traffic_index']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[city].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[city].get(f"speed_val_{condition}", [])
                time_vals = final_dict[city].get(f"time_val_{condition}", [])

                if speed_vals:  # Avoid division by zero or empty dict
                    all_speed_values = [val for inner_dict in speed_vals.values() for val in inner_dict.values()]
                    if all_speed_values:  # Check to avoid computing mean on empty list
                        row_data[f"avg_speed_val_{condition}"] = np.mean(all_speed_values)
                    else:
                        row_data[f"avg_speed_val_{condition}"] = np.nan
                else:
                    row_data[f"avg_speed_val_{condition}"] = np.nan  # Handle empty or missing dict

                if time_vals:
                    all_time_values = [val for inner_dict in time_vals.values() for val in inner_dict.values()]
                    if all_time_values:
                        row_data[f"avg_time_val_{condition}"] = np.mean(all_time_values)
                    else:
                        row_data[f"avg_time_val_{condition}"] = np.nan
                else:
                    row_data[f"avg_time_val_{condition}"] = np.nan  # Handle empty or missing dict

            # Append the row data for the current city
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)

        # Create a new DataFrame to average the columns across conditions
        agg_df = pd.DataFrame()

        # Loop through the columns in the original DataFrame
        for col in df.columns:
            # Extract the feature name (without condition part)
            feature_name = "_".join(col.split("_")[:-1])
            condition = col.split("_")[-1]

            # Create a new column by averaging values across conditions for the same feature
            if feature_name not in agg_df.columns:
                # Select the columns for this feature across all conditions
                condition_cols = [c for c in df.columns if feature_name in c]
                agg_df[feature_name] = df[condition_cols].mean(axis=1)

        # Compute the correlation matrix on the aggregated DataFrame
        corr_matrix_avg = agg_df.corr(method='spearman')

        # Rename the variables in the correlation matrix (example: renaming keys)
        rename_dict_2 = {
            'avg_speed_val': 'Crossing speed', 'avg_time_val': 'Crossing decision time',
            'ped_cross_city': 'Crossing', 'person_city': 'Detected persons',
            'bicycle_city': 'Detected bicycles', 'car_city': 'Detected cars',
            'motorcycle_city': 'Detected motorcycles', 'bus_city': 'Detected bus',
            'truck_city': 'Detected truck', 'cross_evnt_city': 'Crossing without traffic light',
            'vehicle_city': 'Detected total number of motor vehicle', 'cellphone_city': 'Detected cellphone',
            'trf_sign_city': 'Detected traffic signs', 'gmp_city': 'GMP',
            'traffic_mortality_city': 'Traffic mortality', 'literacy_rate_city': 'Literacy rate',
            'gini': 'Gini coefficient', 'traffic_index': 'Traffic Index'
            }

        corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_2, index=rename_dict_2)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation matrix heatmap averaged" # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        plots_class.save_plotly_figure(fig, "correlation_matrix_heatmap_averaged", save_final=True)

        # Continent Wise

        # Initialise a list to store rows of data (one row per city)
        data_rows = []

        # Assuming `conditions` is a list of conditions you are working with
        conditions = ['0', '1']  # Modify this list to include all conditions you have (e.g., '0', '1', etc.)
        unique_continents = df_mapping['continent'].unique()

        # Iterate over each city and condition
        for city in final_dict:
            # Initialise a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'gmp', 'traffic_mortality', 'literacy_rate', 'continent', 'gini', 'traffic_index']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[city].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[city].get(f"speed_val_{condition}", [])
                time_vals = final_dict[city].get(f"time_val_{condition}", [])

                if speed_vals:
                    all_speed_values = [val for inner_dict in speed_vals.values() for val in inner_dict.values()]
                    row_data[f"avg_speed_val_{condition}"] = np.mean(all_speed_values) if all_speed_values else np.nan
                else:
                    row_data[f"avg_speed_val_{condition}"] = np.nan

                # Handle avg_time_val
                if time_vals:
                    all_time_values = [val for inner_dict in time_vals.values() for val in inner_dict.values()]
                    row_data[f"avg_time_val_{condition}"] = np.mean(all_time_values) if all_time_values else np.nan
                else:
                    row_data[f"avg_time_val_{condition}"] = np.nan

            # Append the row data for the current city
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)

        for continents in unique_continents:
            filtered_df = df[(df['continent_0'] == continents) | (df['continent_1'] == continents)]
            # Create a new DataFrame to average the columns across conditions
            agg_df = pd.DataFrame()

            # Loop through the columns in the original DataFrame
            for col in filtered_df.columns:
                # Extract the feature name (without condition part)
                feature_name = "_".join(col.split("_")[:-1])
                condition = col.split("_")[-1]

                # Skip columns named "continent_0" or "continent_1"
                if "continent" in feature_name:
                    continue

                # Create a new column by averaging values across conditions for the same feature
                if feature_name not in agg_df.columns:
                    # Select the columns for this feature across all conditions
                    condition_cols = [c for c in filtered_df.columns if feature_name in c]
                    agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)

            # Compute the correlation matrix on the aggregated DataFrame
            corr_matrix_avg = agg_df.corr(method='spearman')

            # Rename the variables in the correlation matrix (example: renaming keys)
            rename_dict_3 = {
                'avg_speed_val': 'Crossing speed', 'avg_time_val': 'Crossing decision time',
                'ped_cross_city': 'Crossing', 'person_city': 'Detected persons',
                'bicycle_city': 'Detected bicycles', 'car_city': 'Detected cars',
                'motorcycle_city': 'Detected motorcycles', 'bus_city': 'Detected bus',
                'truck_city': 'Detected truck', 'cross_evnt_city': 'Crossing without traffic light',
                'vehicle_city': 'Detected total number of motor vehicle', 'cellphone_city': 'Detected cellphone',
                'trf_sign_city': 'Detected traffic signs', 'gmp': 'GMP',
                'traffic_mortality': 'Traffic mortality', 'literacy_rate': 'Literacy rate', 'gini': 'Gini coefficient',
                'traffic_index': 'Traffic Index'
                }

            corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_3, index=rename_dict_3)

            # Generate the heatmap using Plotly
            fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                            color_continuous_scale='RdBu',  # Color scale (you can customize this)
                            aspect="auto",  # Automatically adjust aspect ratio
                            # title=f"Correlation matrix heatmap {continents}"  # Title of the heatmap
                            )

            fig.update_layout(coloraxis_showscale=False)

            # update font family
            fig.update_layout(font=dict(family=common.get_configs('font_family')))

            # use value from config file
            fig.update_layout(font=dict(size=common.get_configs('font_size')))

            plots_class.save_plotly_figure(fig, f"correlation_matrix_heatmap_{continents}", save_final=True)

    @staticmethod
    def correlation_matrix_country(df_mapping, df_country, save_file=True):
        logger.info("Plotting correlation matrices.")
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        (ped_cross_city, _, person_city, bicycle_city, car_city,
         motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city,
         cellphone_city, trf_sign_city, _, _, _, _, _, avg_speed_country, avg_time_country,
         _, _, _, cross_no_equip_country) = data_tuple[10:33]

        ped_cross_city = wrapper_class.country_sum_from_cities(ped_cross_city, df_mapping)

        person_city = wrapper_class.country_averages_from_nested(person_city, df_mapping)
        bicycle_city = wrapper_class.country_averages_from_nested(bicycle_city, df_mapping)
        car_city = wrapper_class.country_averages_from_nested(car_city, df_mapping)
        motorcycle_city = wrapper_class.country_averages_from_nested(motorcycle_city, df_mapping)
        bus_city = wrapper_class.country_averages_from_nested(bus_city, df_mapping)
        truck_city = wrapper_class.country_averages_from_nested(truck_city, df_mapping)
        vehicle_city = wrapper_class.country_averages_from_nested(vehicle_city, df_mapping)
        cellphone_city = wrapper_class.country_averages_from_nested(cellphone_city, df_mapping)
        trf_sign_city = wrapper_class.country_averages_from_nested(trf_sign_city, df_mapping)

        cross_evnt_city = wrapper_class.country_averages_from_flat(cross_evnt_city, df_mapping)

        # Check if both 'speed' and 'time' are valid dictionaries
        if avg_speed_country is None or avg_time_country is None:
            raise ValueError("Either 'speed' or 'time' returned None, please check the input data or calculations.")

        # Remove the ones where there is data missing for a specific country and condition
        common_keys = avg_speed_country.keys() & avg_time_country.keys()

        # Retain only the key-value pairs where the key is present in both dictionaries
        avg_speed_country = {key: avg_speed_country[key] for key in common_keys}
        avg_time_country = {key: avg_time_country[key] for key in common_keys}

        # Now populate the final_dict with city-wise data
        for country_condition, speed in avg_speed_country.items():
            country, condition = country_condition.split('_')

            # Get the country from the previously stored city_country_map
            iso_code = values_class.get_value(df_mapping, "country", country, None, None, "iso3")
            continent = values_class.get_value(df_mapping, "country", country, None, None, "continent")
            traffic_mortality = values_class.get_value(df_mapping, "country", country, None, None, "traffic_mortality")
            literacy_rate = values_class.get_value(df_mapping, "country", country, None, None, "literacy_rate")
            gini = values_class.get_value(df_mapping, "country", country, None, None, "gini")
            med_age = values_class.get_value(df_mapping, "country", country, None, None, "med_age")
            avg_day_night_speed = values_class.get_value(df_countries, "country", country,
                                                         None, None, "speed_crossing_day_night_country_avg")
            avg_day_night_time = values_class.get_value(df_countries, "country", country,
                                                        None, None, "time_crossing_day_night_country_avg")

            if country or iso_code is not None:

                # Initialise the city's dictionary if not already present
                if f'{country}' not in final_dict:
                    final_dict[f'{country}'] = {
                                                "avg_speed_0": None, "avg_speed_1": None,
                                                "avg_time_0": None, "avg_time_1": None,
                                                "avg_day_night_speed": None, "avg_day_night_time": None,
                                                "ped_cross_city_0": 0, "ped_cross_city_1": 0,
                                                "person_city_0": 0, "person_city_1": 0,
                                                "bicycle_city_0": 0,
                                                "bicycle_city_1": 0,
                                                "car_city_0": 0,
                                                "car_city_1": 0,
                                                "motorcycle_city_0": 0,
                                                "motorcycle_city_1": 0,
                                                "bus_city_0": 0,
                                                "bus_city_1": 0,
                                                "truck_city_0": 0,
                                                "truck_city_1": 0,
                                                "cross_evnt_city_0": 0,
                                                "cross_evnt_city_1": 0,
                                                "vehicle_city_0": 0,
                                                "vehicle_city_1": 0,
                                                "cellphone_city_0": 0,
                                                "cellphone_city_1": 0,
                                                "trf_sign_city_0": 0,
                                                "trf_sign_city_1": 0,
                                                }

                # Populate the corresponding speed and time based on the condition
                final_dict[f'{country}'][f"avg_speed_{condition}"] = speed

                if f'{country}_{condition}' in avg_time_country:
                    final_dict[f'{country}'][f"avg_time_{condition}"] = avg_time_country.get(
                        f'{country}_{condition}', None)
                    final_dict[f'{country}'][f"ped_cross_city_{condition}"] = ped_cross_city.get(
                        f'{country}_{condition}', None)
                    final_dict[f'{country}'][f"person_city_{condition}"] = person_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"bicycle_city_{condition}"] = bicycle_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"car_city_{condition}"] = car_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"motorcycle_city_{condition}"] = motorcycle_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"bus_city_{condition}"] = bus_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"truck_city_{condition}"] = truck_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"cross_evnt_city_{condition}"] = cross_no_equip_country.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"vehicle_city_{condition}"] = vehicle_city.get(
                        f'{country}_{condition}', None)
                    final_dict[f'{country}'][f"cellphone_city_{condition}"] = cellphone_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"trf_sign_city_{condition}"] = trf_sign_city.get(
                        f'{country}_{condition}', 0)

                    final_dict[f'{country}'][f"traffic_mortality_{condition}"] = traffic_mortality
                    final_dict[f'{country}'][f"literacy_rate_{condition}"] = literacy_rate
                    final_dict[f'{country}'][f"gini_{condition}"] = gini
                    final_dict[f'{country}'][f"med_age_{condition}"] = med_age
                    final_dict[f'{country}'][f"continent_{condition}"] = continent

                    final_dict[f'{country}']["avg_day_night_speed"] = avg_day_night_speed
                    final_dict[f'{country}']["avg_day_night_time"] = avg_day_night_time

        # Initialize an empty list to store the rows for the DataFrame
        data_day, data_night = [], []

        # Loop over each city and gather relevant values for condition 0
        for country in final_dict:
            # Initialise a dictionary for the row
            row_day, row_night = {}, {}

            # Add data for condition 0 (ignore 'speed_val' and 'time_val')
            for condition in ['0']:  # Only include condition 0
                for key, value in final_dict[country].items():
                    if (
                        condition in key
                        and 'speed_val' not in key
                        and 'time_val' not in key
                        and 'continent' not in key
                        and 'avg_day_night_speed' not in key
                        and 'avg_day_night_time' not in key
                    ):
                        row_day[key] = value

            # Append the row to the data list
            data_day.append(row_day)

            for condition in ['1']:  # Only include condition 1
                for key, value in final_dict[country].items():
                    if (
                        condition in key
                        and 'speed_val' not in key
                        and 'time_val' not in key
                        and 'continent' not in key
                        and 'avg_day_night_speed' not in key
                        and 'avg_day_night_time' not in key
                    ):
                        row_night[key] = value

            # Append the row to the data list
            data_night.append(row_night)

        # Convert the list of rows into a Pandas DataFrame
        df_day = pd.DataFrame(data_day)
        df_night = pd.DataFrame(data_night)

        # Calculate the correlation matrix
        corr_matrix_day = df_day.corr(method='spearman')
        corr_matrix_night = df_night.corr(method='spearman')

        # Rename the variables in the correlation matrix
        rename_dict_1 = {
            'avg_speed_0': 'Speed', 'avg_speed_1': 'Crossing speed',
            'avg_time_0': 'Crossing decision time', 'avg_time_1': 'Crossing decision time',
            'ped_cross_city_0': 'Crossing', 'ped_cross_city_1': 'Crossing',
            'person_city_0': 'Detected persons', 'person_city_1': 'Detected persons',
            'bicycle_city_0': 'Detected bicycles', 'bicycle_city_1': 'Detected bicycles',
            'car_city_0': 'Detected cars', 'car_city_1': 'Detected cars',
            'motorcycle_city_0': 'Detected motorcycles', 'motorcycle_city_1': 'Detected motorcycles',
            'bus_city_0': 'Detected bus', 'bus_city_1': 'Detected bus',
            'truck_city_0': 'Detected truck', 'truck_city_1': 'Detected truck',
            'cross_evnt_city_0': 'Detected crossings without traffic lights',
            'cross_evnt_city_1': 'Detected crossings without traffic lights',
            'vehicle_city_0': 'Detected motor vehicles',
            'vehicle_city_1': 'Detected motor vehicles',
            'cellphone_city_0': 'Detected cellphones', 'cellphone_city_1': 'Detected cellphones',
            'trf_sign_city_0': 'Detected traffic signs', 'trf_sign_city_1': 'Detected traffic signs',
            'traffic_mortality_0': 'Traffic mortality', 'traffic_mortality_1': 'Traffic mortality',
            'literacy_rate_0': 'Literacy rate', 'literacy_rate_1': 'Literacy rate',
            'gini_0': 'Gini coefficient', 'gini_1': 'Gini coefficient', 'med_age_0': 'Median age',
            'med_age_1': 'Median age',
            }

        corr_matrix_day = corr_matrix_day.rename(columns=rename_dict_1, index=rename_dict_1)
        corr_matrix_night = corr_matrix_night.rename(columns=rename_dict_1, index=rename_dict_1)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_day, text_auto=".2f",  # Display correlation values on the heatmap # type: ignore
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto")  # Automatically adjust aspect ratio
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        plots_class.save_plotly_figure(fig, "correlation_matrix_heatmap_day", save_final=True)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_night, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation Matrix Heatmap in night"  # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        plots_class.save_plotly_figure(fig, "correlation_matrix_heatmap_night", save_final=True)

        # Initialize a list to store rows of data (one row per country)
        data_rows = []

        # Assuming `conditions` is a list of conditions you are working with
        conditions = ['0', '1']  # Modify this list to include all conditions you have (e.g., '0', '1', etc.)

        # Iterate over each country and condition
        for country in final_dict:
            # Initialize a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'traffic_mortality', 'literacy_rate', 'gini', 'med_age']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[country].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[country].get("avg_day_night_speed", [])
                time_vals = final_dict[country].get("avg_day_night_time", [])

                if speed_vals:  # Avoid division by zero or empty arrays
                    row_data["avg_day_night_speed"] = np.mean(speed_vals)
                else:
                    row_data["avg_day_night_speed"] = np.nan  # Handle empty or missing arrays

                if time_vals:
                    row_data["avg_day_night_time"] = np.mean(time_vals)
                else:
                    row_data["avg_day_night_time"] = np.nan  # Handle empty or missing arrays

            # Append the row data for the current country
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)
        # df = df[[col for col in df.columns if col.endswith("_0") or col.endswith("_1")]]

        # Create a new DataFrame to average the columns across conditions
        agg_df = pd.DataFrame()

        # Define known conditions (from earlier)
        conditions = ['0', '1']

        for col in df.columns:
            # Check if the column ends with a known condition
            if any(col.endswith(f"_{cond}") for cond in conditions):
                feature_name = "_".join(col.split("_")[:-1])
                if feature_name not in agg_df.columns:
                    condition_cols = [c for c in df.columns if c.startswith(feature_name + "_")]
                    agg_df[feature_name] = df[condition_cols].mean(axis=1)
            else:
                # Directly copy columns that don't follow the condition pattern (like avg_day_night_speed)
                agg_df[col] = df[col]

            # Create a new column by averaging values across conditions for the same feature
            if feature_name not in agg_df.columns:
                # Select the columns for this feature across all conditions
                condition_cols = [c for c in df.columns if c.startswith(feature_name + "_")]  # type: ignore
                agg_df[feature_name] = df[condition_cols].mean(axis=1)

        # Compute the correlation matrix on the aggregated DataFrame
        corr_matrix_avg = agg_df.corr(method='spearman')

        # Rename the variables in the correlation matrix (example: renaming keys)
        rename_dict_2 = {
            'avg_day_night_speed': 'Crossing speed', 'avg_day_night_time': 'Crossing decision time',
            'ped_cross_city': 'Crossing', 'person_city': 'Detected persons',
            'bicycle_city': 'Detected bicycles', 'car_city': 'Detected cars',
            'motorcycle_city': 'Detected motorcycles', 'bus_city': 'Detected bus',
            'truck_city': 'Detected truck', 'cross_evnt_city': 'Crossing without traffic light',
            'vehicle_city': 'Detected total number of motor vehicle', 'cellphone_city': 'Detected cellphone',
            'trf_sign_city': 'Detected traffic signs',
            'traffic_mortality': 'Traffic mortality', 'literacy_rate': 'Literacy rate',
            'gini': 'Gini coefficient', 'med_age': 'Median age'
            }

        corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_2, index=rename_dict_2)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation matrix heatmap averaged" # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        fig.update_traces(textfont_size=14)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=18))
        fig.update_yaxes(tickangle=0, tickfont=dict(size=18))

        plots_class.save_plotly_figure(fig, "correlation_matrix_heatmap_averaged", save_final=True)

        # Continent Wise

        # Initialize a list to store rows of data (one row per country)
        data_rows = []

        # Assuming `conditions` is a list of conditions you are working with
        conditions = ['0', '1']  # Modify this list to include all conditions you have (e.g., '0', '1', etc.)
        unique_continents = df_mapping['continent'].unique()

        # Iterate over each country and condition
        for country in final_dict:
            # Initialize a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'traffic_mortality', 'literacy_rate', 'continent', 'gini', 'med_age']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[country].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[country].get("avg_day_night_speed", [])
                time_vals = final_dict[country].get("avg_day_night_time", [])

                if speed_vals:  # Avoid division by zero or empty arrays
                    row_data["avg_day_night_speed"] = np.mean(speed_vals)
                else:
                    row_data["avg_day_night_speed"] = np.nan  # Handle empty or missing arrays

                if time_vals:
                    row_data["avg_day_night_time"] = np.mean(time_vals)
                else:
                    row_data["avg_day_night_time"] = np.nan  # Handle empty or missing arrays

            # Append the row data for the current country
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)

        for continents in unique_continents:
            filtered_df = df[(df['continent_0'] == continents) | (df['continent_1'] == continents)]
            # Create a new DataFrame to average the columns across conditions
            agg_df = pd.DataFrame()

            # Define known conditions (from earlier)
            conditions = ['0', '1']

            for col in filtered_df.columns:
                # Check if the column ends with a known condition
                if any(col.endswith(f"_{cond}") for cond in conditions):
                    feature_name = "_".join(col.split("_")[:-1])
                    # Skip columns named "continent_0" or "continent_1"
                    if "continent" in feature_name:
                        continue
                    if feature_name not in agg_df.columns:
                        condition_cols = [c for c in filtered_df.columns if c.startswith(feature_name + "_")]
                        if all(pd.api.types.is_numeric_dtype(filtered_df[c]) for c in condition_cols):
                            agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)
                        else:
                            logger.debug(f"Skipping non-numeric feature: {feature_name}")

                else:
                    agg_df[col] = filtered_df[col]

                # Create a new column by averaging values across conditions for the same feature
                if feature_name not in agg_df.columns:
                    # Select the columns for this feature across all conditions
                    condition_cols = [c for c in filtered_df.columns if feature_name in c]
                    agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)

            # Compute the correlation matrix on the aggregated DataFrame
            corr_matrix_avg = agg_df.corr(method='spearman')

            # Rename the variables in the correlation matrix (example: renaming keys)
            rename_dict_3 = {
                'avg_day_night_speed': 'Crossing speed', "avg_day_night_time": 'Crossing decision time',
                'ped_cross_city': 'Crossing', 'person_city': 'Detected persons',
                'bicycle_city': 'Detected bicycles', 'car_city': 'Detected cars',
                'motorcycle_city': 'Detected motorcycles', 'bus_city': 'Detected bus',
                'truck_city': 'Detected truck', 'cross_evnt_city': 'Crossing without traffic light',
                'vehicle_city': 'Detected total number of motor vehicle', 'cellphone_city': 'Detected cellphone',
                'trf_sign_city': 'Detected traffic signs',
                'traffic_mortality': 'Traffic mortality', 'literacy_rate': 'Literacy rate', 'gini': 'Gini coefficient',
                'med_age': 'Median age'
                }

            corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_3, index=rename_dict_3)

            # Generate the heatmap using Plotly
            fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                            color_continuous_scale='RdBu',  # Color scale (you can customize this)
                            aspect="auto",  # Automatically adjust aspect ratio
                            # title=f"Correlation matrix heatmap {continents}"  # Title of the heatmap
                            )

            fig.update_layout(coloraxis_showscale=False)

            # update font family
            fig.update_layout(font=dict(family=common.get_configs('font_family')))

            # use value from config file
            fig.update_layout(font=dict(size=common.get_configs('font_size')))

            # Update text font size inside heatmap
            fig.update_traces(textfont_size=14)
            fig.update_xaxes(tickangle=45, tickfont=dict(size=18))
            fig.update_yaxes(tickangle=0, tickfont=dict(size=18))

            # save file to local output folder
            if save_file:
                # Final adjustments and display
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                plots_class.save_plotly_figure(fig, f"correlation_matrix_heatmap_{continents}", save_final=True)
            # open it in localhost instead
            else:
                fig.show()

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
        df = df.drop(columns=['id', 'city', 'state', 'lat', 'lon', 'gmp', 'population_city', 'traffic_index',
                              'upload_date', 'speed_crossing_day_city', 'speed_crossing_night_city',
                              'speed_crossing_day_night_city_avg', 'time_crossing_day_city',
                              'time_crossing_night_city', 'time_crossing_day_night_city_avg',
                              'with_trf_light_day_city', 'with_trf_light_night_city',
                              'without_trf_light_day_city', 'without_trf_light_night_city',
                              'channel'], errors='ignore')

        static_columns = [
            'country', 'continent', 'population_country', 'traffic_mortality',
            'literacy_rate', 'avg_height', 'gini', 'med_age', 'speed_crossing_day_country',
            'speed_crossing_night_country', 'speed_crossing_day_night_country_avg',
            'time_crossing_day_country', 'time_crossing_night_country', 'time_crossing_day_night_country_avg',
            'with_trf_light_day_country', 'with_trf_light_night_country', 'without_trf_light_day_country',
            'without_trf_light_night_country'
            ]

        # Columns to merge as lists
        merge_columns = ['videos', 'time_of_day', 'start_time', 'end_time', 'vehicle_type', 'fps_list']

        sum_columns = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'cellphone', 'traffic_light', 'stop_sign', 'total_time', 'total_videos'
        ]

        agg_dict = {col: 'first' for col in static_columns}
        agg_dict.update({col: lambda x: list(x) for col in merge_columns})  # type: ignore
        agg_dict.update({col: 'sum' for col in sum_columns})

        # Fix continent assignment - assign most frequent continent per iso3
        continent_mode = (
            df.groupby('iso3')['continent']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            .reset_index()
            .rename(columns={'continent': 'continent_majority'})
        )

        # Merge back to df to fix 'continent' column
        df = df.drop('continent', axis=1)
        df = df.merge(continent_mode, on='iso3', how='left')
        df = df.rename(columns={'continent_majority': 'continent'})

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

                                # Load tracking DataFrame for the current video segment
                                for folder_path in common.get_configs('data'):
                                    for file in os.listdir(folder_path):
                                        if video_start_time in file and file.endswith('.csv'):
                                            file_path = os.path.join(folder_path, file)
                                            # Load the CSV
                                            df = pd.read_csv(file_path)
                                            break
                                filtered_df = df[df['Unique Id'] == unique_id]

                                if filtered_df.empty:
                                    return None, None  # No data found for this unique_id

                                # Determine frame-based start and end times
                                first_frame = filtered_df['Frame Count'].min()
                                last_frame = filtered_df['Frame Count'].max()

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

    def clean_csv_filename(self, file):
        """
        If the filename ends with '.csv', returns it as-is.
        Otherwise:
        - Removes leading dot
        - Truncates at first '.csv' if present
        - Else returns cleaned filename
        """
        if file.endswith('.csv'):
            return file
        file_clean = file.lstrip('.')  # Remove leading dot if present
        csv_pos = file_clean.find('.csv')
        if csv_pos != -1:
            base_name = file_clean[:csv_pos + 4]  # includes ".csv"
        else:
            base_name = file_clean  # fallback if '.csv' not found
        return base_name

    def add_speed_and_time_to_mapping(self, df_mapping, avg_speed_city=None, avg_time_city=None,
                                      avg_speed_country=None, avg_time_country=None):
        """
        Adds city/country-level average speeds and/or times (day/night) to df_mapping DataFrame,
        depending on which dicts are provided. Missing columns are created and initialized with NaN.
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

            # Ensure columns exist and are initialized to np.nan
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
                    country, time_of_day = parts[0], int(parts[1])
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
             pedestrian_cross_country                       # 35
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

        # Data to avoid showing on hover in scatter plots
        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type',
                          'channel']
        hover_data = list(set(df.columns) - set(columns_remove))

        # mapbox map with all data
        plots_class.mapbox_map(df=df, hover_data=hover_data, file_name='mapbox_map_all')
        plots_class.mapbox_map(df=df,
                               hover_data=hover_data,
                               density_col='population_city',
                               density_radius=10,
                               file_name='mapbox_map_all_pop')

        # Get the population threshold from the configuration
        population_threshold = common.get_configs("population_threshold")

        # Get the minimum percentage of country population from the configuration
        min_percentage = common.get_configs("min_city_population_percentage")

        # Filter df_mapping to include cities that meet either of the following criteria:
        # 1. The city's population is greater than the threshold
        # 2. The city's population is at least the minimum percentage of the country's population
        df_mapping = df_mapping[
            (df_mapping["population_city"] >= population_threshold) |  # Condition 1
            (df_mapping["population_city"] >= min_percentage * df_mapping["population_country"])  # Condition 2
        ]

        # Limit countries if required
        countries_include = common.get_configs("countries_analyse")
        if countries_include:
            df_mapping = df_mapping[df_mapping["iso3"].isin(common.get_configs("countries_analyse"))]

        person_counter, bicycle_counter, car_counter, motorcycle_counter = 0, 0, 0, 0
        bus_counter, truck_counter, cellphone_counter, traffic_light_counter, stop_sign_counter = 0, 0, 0, 0, 0

        total_duration = Analysis.calculate_total_seconds(df_mapping)
        logger.info(f"Duration of videos in seconds: {total_duration}, in minutes: {total_duration/60:.2f}, in " +
                    f"hours: {total_duration/60/60:.2f}.")
        logger.info("Total number of videos: {}.", Analysis.calculate_total_videos(df_mapping))
        country, number = Analysis.get_unique_values(df_mapping, "iso3")
        logger.info("Total number of countries and territories: {}.", number)

        city, number = Analysis.get_unique_values(df_mapping, "city")
        logger.info("Total number of cities: {}.", number)

        # add information for each city to then be appended to mapping
        df_mapping['person'] = 0
        df_mapping['bicycle'] = 0
        df_mapping['car'] = 0
        df_mapping['motorcycle'] = 0
        df_mapping['bus'] = 0
        df_mapping['truck'] = 0
        df_mapping['cellphone'] = 0
        df_mapping['traffic_light'] = 0
        df_mapping['stop_sign'] = 0
        df_mapping['total_time'] = 0
        df_mapping['total_crossing_detect'] = 0

        # City-level columns
        df_mapping['speed_crossing_day_city'] = math.nan
        df_mapping['speed_crossing_night_city'] = math.nan
        df_mapping['speed_crossing_day_night_city_avg'] = math.nan

        df_mapping['time_crossing_day_city'] = math.nan
        df_mapping['time_crossing_night_city'] = math.nan
        df_mapping['time_crossing_day_night_city_avg'] = math.nan

        df_mapping['with_trf_light_day_city'] = 0.0
        df_mapping['with_trf_light_night_city'] = 0.0

        df_mapping['without_trf_light_day_city'] = 0.0
        df_mapping['without_trf_light_night_city'] = 0.0

        # Country-level columns
        df_mapping['speed_crossing_day_country'] = math.nan
        df_mapping['speed_crossing_night_country'] = math.nan
        df_mapping['speed_crossing_day_night_country_avg'] = math.nan

        df_mapping['time_crossing_day_country'] = math.nan
        df_mapping['time_crossing_night_country'] = math.nan
        df_mapping['time_crossing_day_night_country_avg'] = math.nan

        df_mapping['with_trf_light_day_country'] = 0.0
        df_mapping['with_trf_light_night_country'] = 0.0

        df_mapping['without_trf_light_day_country'] = 0.0
        df_mapping['without_trf_light_night_country'] = 0.0

        all_speed = {}
        all_time = {}

        logger.info("Processing csv files.")
        pedestrian_crossing_count, data = {}, {}
        for folder_path in common.get_configs('data'):
            if not os.path.exists(folder_path):
                logger.warning(f"Folder does not exist: {folder_path}.")
                continue

            for file in tqdm(os.listdir(folder_path), desc=f"Processing files in {folder_path}"):
                file = analysis_class.filter_csv_files(file=file, df_mapping=df_mapping)
                # list of misc and trash files
                misc_files = ["DS_Store" , "seg", "bbox"]
                if file is None or file in misc_files:  # exclude not useful files
                    continue
                else:
                    filename_no_ext = os.path.splitext(file)[0]
                    logger.debug(f"{filename_no_ext}: fetching values.")

                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    # After reading the file, clean up the filename
                    base_name = analysis_class.clean_csv_filename(file)

                    filename_no_ext = os.path.splitext(base_name)[0]  # Remove extension

                    video_id, start_index = filename_no_ext.rsplit("_", 1)  # split to extract id and index

                    video_city_id = Analysis.find_city_id(df_mapping, video_id, int(start_index))
                    video_city = df_mapping.loc[df_mapping["id"] == video_city_id, "city"].values[0]
                    video_state = df_mapping.loc[df_mapping["id"] == video_city_id, "state"].values[0]
                    video_country = df_mapping.loc[df_mapping["id"] == video_city_id, "country"].values[0]

                    logger.debug(f"{file}: found values {video_city}, {video_state}, {video_country}.")

                    # Get the number of number and unique id of the object crossing the road
                    ids = algorithms_class.pedestrian_crossing(df,
                                                               filename_no_ext,
                                                               df_mapping,
                                                               common.get_configs("boundary_left"),
                                                               common.get_configs("boundary_right"),
                                                               person_id=0)

                    # Saving it in a dictionary in: {video-id_time: count, ids}
                    pedestrian_crossing_count[filename_no_ext] = {"ids": ids}

                    # Saves the time to cross in form {name_time: {id(s): time(s)}}
                    temp_data = algorithms_class.time_to_cross(df,
                                                               pedestrian_crossing_count[filename_no_ext]["ids"],
                                                               filename_no_ext,
                                                               df_mapping)
                    data[filename_no_ext] = temp_data

                    # Calculate the total number of different objects detected
                    person_video = Analysis.count_object(df, 0)
                    person_counter += person_video
                    df_mapping.loc[df_mapping["id"] == video_city_id, "person"] += person_video

                    bicycle_video = Analysis.count_object(df, 1)
                    bicycle_counter += bicycle_video
                    df_mapping.loc[df_mapping["id"] == video_city_id, "bicycle"] += bicycle_video

                    car_video = Analysis.count_object(df, 2)
                    car_counter += car_video
                    df_mapping.loc[df_mapping["id"] == video_city_id, "car"] += car_video

                    motorcycle_video = Analysis.count_object(df, 3)
                    motorcycle_counter += motorcycle_video
                    df_mapping.loc[df_mapping["id"] == video_city_id, "motorcycle"] += motorcycle_video

                    bus_video = Analysis.count_object(df, 5)
                    bus_counter += bus_video
                    df_mapping.loc[df_mapping["id"] == video_city_id, "bus"] += bus_video

                    truck_video = Analysis.count_object(df, 7)
                    truck_counter += truck_video
                    df_mapping.loc[df_mapping["id"] == video_city_id, "truck"] += truck_video

                    cellphone_video = Analysis.count_object(df, 67)
                    cellphone_counter += cellphone_video
                    df_mapping.loc[df_mapping["id"] == video_city_id, "cellphone"] += cellphone_video

                    traffic_light_video = Analysis.count_object(df, 9)
                    traffic_light_counter += traffic_light_video
                    df_mapping.loc[df_mapping["id"] == video_city_id, "traffic_light"] += traffic_light_video

                    stop_sign_video = Analysis.count_object(df, 11)
                    stop_sign_counter += stop_sign_video
                    df_mapping.loc[df_mapping["id"] == video_city_id, "stop_sign"] += stop_sign_video

                    # add duration of segment
                    time_video = analysis_class.get_duration(df_mapping, video_id, int(start_index))
                    df_mapping.loc[df_mapping["id"] == video_city_id, "total_time"] += time_video  # type: ignore

                    # add total crossing detected
                    df_mapping.loc[df_mapping["id"] == video_city_id, "total_crossing_detect"] += len(ids)

                    # Aggregated values
                    speed_value = algorithms_class.calculate_speed_of_crossing(df_mapping,
                                                                               df,
                                                                               {filename_no_ext: temp_data})

                    if speed_value is not None:
                        all_speed.update(speed_value)

                    time_value = algorithms_class.time_to_start_cross(df_mapping,
                                                                      df,
                                                                      {filename_no_ext: temp_data})

                    if time_value is not None:
                        all_time.update(time_value)

        # Record the average speed and time of crossing on country basis
        avg_speed_country = algorithms_class.avg_speed_of_crossing_country(df_mapping, all_speed)
        avg_time_country = algorithms_class.avg_time_to_start_cross_country(df_mapping, all_speed)

        # Record the average speed and time of crossing on city basis
        avg_speed_city = algorithms_class.avg_speed_of_crossing_city(all_speed)
        avg_time_city = algorithms_class.avg_time_to_start_cross_city(df_mapping, all_time)

        # Kill the program if there is no data to analyse
        if len(avg_time_city) == 0 or len(avg_speed_city) == 0:
            logger.error("No speed and time data to analyse.")
            exit()

        df_mapping = analysis_class.add_speed_and_time_to_mapping(df_mapping=df_mapping,
                                                                  avg_speed_city=avg_speed_city,
                                                                  avg_speed_country=avg_speed_country,
                                                                  avg_time_city=avg_time_city,
                                                                  avg_time_country=avg_time_country)

        min_max_speed = analysis_class.get_duration_segment(all_speed, df_mapping, name="speed", duration=None)
        min_max_time = analysis_class.get_duration_segment(all_time, df_mapping, name="time", duration=None)

        logger.info("Calculating counts of detected traffic signs.")
        traffic_sign_city = analysis_class.calculate_traffic_signs(df_mapping)
        logger.info("Calculating counts of detected mobile phones.")
        cellphone_city = Analysis.calculate_cell_phones(df_mapping)
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
        pedestrian_cross_city = Analysis.pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping)
        logger.info("Calculating counts of crossing events in countries.")
        pedestrian_cross_country = Analysis.pedestrian_cross_per_country(pedestrian_cross_city, df_mapping)

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
                                   "state"].iloc[0] if "state" in df_mapping.columns else None

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
                                   "state"].iloc[0] if "state" in df_mapping.columns else None

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
                         pedestrian_cross_country),                         # 35
                        file)
        logger.info("Analysis results saved to pickle file.")

    # Set index as ID
    df_mapping = df_mapping.set_index("id", drop=False)

    # --- Check if reanalysis of speed is required ---
    if common.get_configs("reanalyse_speed"):
        # Compute average speed for each country using mapping and speed data
        avg_speed_country = algorithms_class.avg_speed_of_crossing_country(df_mapping, all_speed)
        # Compute average speed for each city using speed data
        avg_speed_city = algorithms_class.avg_speed_of_crossing_city(all_speed)

        # Add computed speed values to the main mapping dataframe
        df_mapping = analysis_class.add_speed_and_time_to_mapping(
            df_mapping=df_mapping,
            avg_speed_city=avg_speed_city,
            avg_speed_country=avg_speed_country
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
            avg_time_country=avg_time_country
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
        # --- Remove low-detection countries from country-level speed/time ---
        # Find countries with crossings below threshold
        remove_countries = {country for country, value in pedestrian_cross_country.items()
                            if value < common.get_configs("min_crossing_detect")}

        # Remove rows from df_mapping where 'country' is in remove_countries
        df_mapping = df_mapping[~df_mapping['country'].isin(remove_countries)].copy()

        # Remove all entries in avg_speed_country and avg_time_country for those countries
        for dict_name, d in [('avg_speed_country', avg_speed_country), ('avg_time_country', avg_time_country)]:
            keys_to_remove = [key for key in d if key.split('_')[0] in remove_countries]
            for key in keys_to_remove:
                logger.info(f"Deleting from {dict_name}: {key} -> {d[key]}")
                del d[key]

        # --- Remove low-detection cities from city-level speed/time ---
        # Sum all conditions for each city in pedestrian_cross_city
        city_sum = defaultdict(int)
        for key, value in pedestrian_cross_city.items():
            city = key.split('_')[0]
            city_sum[city] += value

        # Find cities with total crossings below threshold
        remove_cities = {city for city, total in city_sum.items()
                         if total < common.get_configs("min_crossing_detect")}

        # Remove rows from df_mapping where 'cities' is in remove_cities
        df_mapping = df_mapping[~df_mapping['city'].isin(remove_cities)].copy()

        # Remove all entries in avg_speed_city and avg_time_city for those cities
        for dict_name, d in [('avg_speed_city', avg_speed_city), ('avg_time_city', avg_time_city)]:
            keys_to_remove = [key for key in d if key.split('_')[0] in remove_cities]
            for key in keys_to_remove:
                logger.info(f"Deleting from {dict_name}: {key} -> {d[key]}")
                del d[key]

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

    # Maps
    plots_class.mapbox_map(df=df, hover_data=hover_data, file_name='mapbox_map')
    plots_class.mapbox_map(df=df,
                           hover_data=hover_data,
                           density_col='total_time',
                           density_radius=10,
                           file_name='mapbox_map_time')
    plots_class.world_map(df_mapping=df)  # map with countries

    plots_class.violin_plot(data_index=22, name="speed", min_threshold=common.get_configs("min_speed_limit"),
                            max_threshold=common.get_configs("max_speed_limit"), df_mapping=df_mapping, save_file=True)

    plots_class.hist(data_index=22, name="speed", min_threshold=common.get_configs("min_speed_limit"),
                     max_threshold=common.get_configs("max_speed_limit"), save_file=True)

    plots_class.hist(data_index=23, name="time", min_threshold=common.get_configs("min_waiting_time"),
                     max_threshold=common.get_configs("max_waiting_time"), save_file=True)

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
                               order_by="average",
                               metric="num_of_crossing",
                               data_view="day",
                               title_text="Crossing in the country",
                               filename="crossing_country",
                               font_size_captions=common.get_configs("font_size") + 8,
                               legend_x=0.87,
                               legend_y=0.04,
                               legend_spacing=0.02
                               )

        plots_class.stack_plot(df,
                               order_by="alphabetical",
                               metric="time",
                               data_view="combined",
                               title_text="Time to start crossing (s)",
                               filename="time_crossing_alphabetical",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=80,
                               right_margin=80
                               )

        plots_class.stack_plot(df,
                               order_by="alphabetical",
                               metric="time",
                               data_view="day",
                               title_text="Time to start crossing (s)",
                               filename="time_crossing_alphabetical_day",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=80,
                               right_margin=80
                               )

        plots_class.stack_plot(df,
                               order_by="alphabetical",
                               metric="time",
                               data_view="night",
                               title_text="Time to start crossing (s)",
                               filename="time_crossing_alphabetical_night",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=80,
                               right_margin=80
                               )

        plots_class.stack_plot(df,
                               order_by="average",
                               metric="time",
                               data_view="combined",
                               title_text="Time to start crossing (s)",
                               filename="time_crossing_avg",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=10,
                               right_margin=10
                               )

        plots_class.stack_plot(df,
                               order_by="average",
                               metric="time",
                               data_view="day",
                               title_text="Time to start crossing (s)",
                               filename="time_crossing_avg_day",
                               font_size_captions=common.get_configs("font_size") + 8,
                               left_margin=10,
                               right_margin=10
                               )

        plots_class.stack_plot(df,
                               order_by="average",
                               metric="time",
                               data_view="night",
                               title_text="Time to start crossing (s)",
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

        Analysis.correlation_matrix(df_mapping)

        # Speed of crossing vs time to start crossing
        df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
        df = df[df["time_crossing"] != 0]
        df['state'] = df['state'].fillna('NA')
        plots_class.scatter(df=df,
                            x="speed_crossing",
                            y="time_crossing",
                            color="continent",
                            text="city",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Mean time to start crossing (in s)',
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
                            yaxis_title='Crossing decision time during daytime (in s)',
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
                            yaxis_title='Crossing decision time during night time (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
        Analysis.plot_crossing_without_traffic_light(df_mapping,
                                                     x_axis_title_height=60,
                                                     font_size_captions=common.get_configs("font_size"),
                                                     legend_x=0.97,
                                                     legend_y=1.0,
                                                     legend_spacing=0.004)
        Analysis.plot_crossing_with_traffic_light(df_mapping,
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
        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type']
        hover_data = list(set(df_countries.columns) - set(columns_remove))
        df_countries.to_csv(os.path.join(common.output_dir, "mapping_updated_countries.csv"))

        # Map with images. currently works on a 13" MacBook air screen in chrome, as things are hardcoded...
        plots_class.map_political(df=df_countries, df_mapping=df_mapping, show_cities=True, show_images=True,
                                  hover_data=hover_data, save_file=True, save_final=False)
        # Map with no images
        plots_class.map_political(df=df_countries, df_mapping=df_mapping, show_cities=True, show_images=False,
                                  hover_data=hover_data, save_file=True, save_final=True)

        # Amount of footage
        plots_class.scatter(df=df_countries,
                            x="total_time",
                            y="person",
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
                                       title_text="Time to start crossing (s)",
                                       filename="time_crossing_avg",
                                       font_size_captions=common.get_configs("font_size") + 8,
                                       legend_x=0.87,
                                       legend_y=0.04,
                                       legend_spacing=0.02)

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="time",
                                       data_view="combined",
                                       title_text="Time to start crossing (s)",
                                       filename="time_crossing_alphabetical",
                                       font_size_captions=common.get_configs("font_size"),
                                       legend_x=0.94,
                                       legend_y=0.03,
                                       legend_spacing=0.02)

        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="speed",
                                       data_view="combined",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_avg",
                                       font_size_captions=common.get_configs("font_size") + 8,
                                       legend_x=0.87,
                                       legend_y=0.04,
                                       legend_spacing=0.02)

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="speed",
                                       data_view="combined",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_alphabetical",
                                       font_size_captions=common.get_configs("font_size"),
                                       legend_x=0.94,
                                       legend_y=0.03,
                                       legend_spacing=0.02)

        # Plotting stacked plot during day
        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="time",
                                       data_view="day",
                                       title_text="Time to start crossing (s)",
                                       filename="time_crossing_avg_day",
                                       font_size_captions=common.get_configs("font_size"))

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="time",
                                       data_view="day",
                                       title_text="Time to start crossing (s)",
                                       filename="time_crossing_alphabetical_day",
                                       font_size_captions=common.get_configs("font_size"))

        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="speed",
                                       data_view="day",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_avg_day",
                                       font_size_captions=common.get_configs("font_size"))

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="speed",
                                       data_view="day",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_alphabetical_day",
                                       font_size_captions=common.get_configs("font_size"))

        # Plotting stacked plot during night
        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="time",
                                       data_view="night",
                                       title_text="Time to start crossing (s)",
                                       filename="time_crossing_avg_night",
                                       font_size_captions=common.get_configs("font_size"))

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="time",
                                       data_view="night",
                                       title_text="Time to start crossing (s)",
                                       filename="time_crossing_alphabetical_night",
                                       font_size_captions=common.get_configs("font_size"))

        plots_class.stack_plot_country(df_countries,
                                       order_by="average",
                                       metric="speed",
                                       data_view="night",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_avg_night",
                                       font_size_captions=common.get_configs("font_size"))

        plots_class.stack_plot_country(df_countries,
                                       order_by="alphabetical",
                                       metric="speed",
                                       data_view="night",
                                       title_text="Mean speed of crossing (in m/s)",
                                       filename="crossing_speed_alphabetical_night",
                                       font_size_captions=common.get_configs("font_size"))

        plots_class.speed_and_time_to_start_cross_country(df_countries,
                                                          x_axis_title_height=110,
                                                          font_size_captions=common.get_configs("font_size") + 8,
                                                          legend_x=0.87,
                                                          legend_y=0.04,
                                                          legend_spacing=0.01)

        Analysis.correlation_matrix_country(df_mapping, df_countries)

        # Speed of crossing vs time to start crossing
        df = df_countries[df_countries["speed_crossing_day_night_country_avg"] != 0].copy()
        df = df[df["time_crossing_day_night_country_avg"] != 0]
        plots_class.scatter(df=df,
                            x="speed_crossing_day_night_country_avg",
                            y="time_crossing_day_night_country_avg",
                            color="continent",
                            text="iso3",
                            xaxis_title='Mean speed of crossing (in m/s)',
                            yaxis_title='Mean time to start crossing (in s)',
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
                            yaxis_title='Crossing decision time during daytime (in s)',
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
                            yaxis_title='Crossing decision time during night time (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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
                            xaxis_title='Mean time to start crossing (in s)',
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

        # Maps with heatmaps
        plots_class.map(df_countries, 'speed_crossing_day_night_country_avg',
                        "Mean speed of crossing (in m/s)", save_file=True)
        plots_class.map(df_countries, 'time_crossing_day_night_country_avg',
                        "Mean time to start crossing (in s)", save_file=True)

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
