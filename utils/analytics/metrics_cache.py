import pandas as pd
import os
import ast
import re
from tqdm import tqdm
import common
from typing import Dict, Any, ClassVar
from custom_logger import CustomLogger
from utils.core.metadata import MetaData
from utils.core.grouping import Grouping

metadata_class = MetaData()
grouping_class = Grouping()
logger = CustomLogger(__name__)  # use custom logger


class Metrics_cache:
    def __init__(self) -> None:
        pass

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
                    result = metadata_class.find_values_with_video_id(df_mapping, f"{vid}_{start_time}_{fps}")
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
                    # Keep only rows with confidence > min_conf
                    dataframe = dataframe[dataframe["confidence"] >= common.get_configs("min_confidence")]

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
            wrapped = grouping_class.city_country_wrapper(
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

    def get_unique_values(self, df, value):
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
