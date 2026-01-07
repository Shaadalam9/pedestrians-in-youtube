import os
import ast
import re
from tqdm import tqdm
import common
from typing import Dict, Any, ClassVar

import polars as pl

from custom_logger import CustomLogger
from utils.core.metadata import MetaData
from utils.core.grouping import Grouping

metadata_class = MetaData()
grouping_class = Grouping()
logger = CustomLogger(__name__)  # use custom logger


class Metrics_cache:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _parse_videos_cell(videos_cell: str | None) -> list[str]:
        """
        Extract video ids robustly from strings like:
          [abc]
          "[abc,def]"
          ["abc","def"]
          []
        """
        if not isinstance(videos_cell, str):
            return []
        # same intent as your original: extract tokens like ids (letters/digits/_/-)
        return re.findall(r"[\w-]+", videos_cell)

    @classmethod
    def _compute_all_metrics(cls, df_mapping: pl.DataFrame) -> None:
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
        data_folders = common.get_configs("data")
        csv_files: Dict[str, str] = {}

        # Index all CSV files from bbox and seg subfolders for quick lookup
        for folder_path in data_folders:
            for subfolder in common.get_configs("sub_domain"):
                subfolder_path = os.path.join(folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    continue
                for file in os.listdir(subfolder_path):
                    if file.endswith(".csv"):
                        csv_files[file] = os.path.join(subfolder_path, file)

        cellphone_info: Dict[str, float] = {}
        traffic_signs_layer: Dict[str, float] = {}
        vehicle_layer: Dict[str, float] = {}
        bicycle_layer: Dict[str, float] = {}
        car_layer: Dict[str, float] = {}
        motorcycle_layer: Dict[str, float] = {}
        bus_layer: Dict[str, float] = {}
        truck_layer: Dict[str, float] = {}
        person_layer: Dict[str, float] = {}

        min_conf = common.get_configs("min_confidence")

        # Iterate mapping rows (Polars)
        mapping_iter = df_mapping.select(["videos", "start_time", "time_of_day"]).iter_rows(named=True)

        for row in tqdm(mapping_iter, total=df_mapping.height, desc="Analysing the csv files:"):
            videos_cell = row.get("videos")
            start_time_cell = row.get("start_time")
            time_of_day_cell = row.get("time_of_day")

            video_ids = cls._parse_videos_cell(videos_cell)

            try:
                start_times = ast.literal_eval(start_time_cell) if isinstance(start_time_cell, str) else None
                time_of_day = ast.literal_eval(time_of_day_cell) if isinstance(time_of_day_cell, str) else None
            except Exception:
                continue

            if not (isinstance(start_times, list) and isinstance(time_of_day, list)):
                continue

            # Loop through all video_id + start_time pairs
            for vid, start_times_list, time_of_day_list in zip(video_ids, start_times, time_of_day):
                if not isinstance(start_times_list, list) or not isinstance(time_of_day_list, list):
                    continue

                for start_time, time_of_day_value in zip(start_times_list, time_of_day_list):
                    prefix = f"{vid}_{start_time}_"

                    matching_files = [
                        fname for fname in csv_files
                        if fname.startswith(prefix) and fname.endswith(".csv")
                    ]
                    if not matching_files:
                        logger.warning(f"[WARNING] File not found for prefix: {prefix}")
                        continue
                    elif len(matching_files) > 1:
                        logger.warning(
                            f"[WARNING] Multiple files found for prefix: {prefix}, using the first one: {matching_files[0]}"  # noqa: E501
                        )

                    filename = matching_files[0]

                    # Extract fps using regex
                    match = re.match(rf"{re.escape(str(vid))}_{re.escape(str(start_time))}_(\d+)\.csv", filename)
                    if match:
                        fps = int(match.group(1))
                    else:
                        logger.error(f"[ERROR] Could not extract fps from filename: {filename}")
                        continue

                    filename = f"{vid}_{start_time}_{fps}.csv"
                    if filename not in csv_files:
                        continue

                    file_path = csv_files[filename]

                    # Find video meta details (start, end, city, location, etc.)
                    # Assumes MetaData helper accepts Polars df_mapping (as per your request)
                    result = metadata_class.find_values_with_video_id(df_mapping, f"{vid}_{start_time}_{fps}")
                    if result is None:
                        continue

                    start = result[1]
                    end = result[2]
                    fps = result[17]
                    duration = end - start  # seconds
                    video_key = f"{vid}_{start_time}_{fps}"

                    # Load detection data for this video segment (Polars)
                    try:
                        dataframe = pl.read_csv(file_path)
                    except Exception as e:
                        logger.warning(f"[WARNING] Failed reading {file_path}: {e}")
                        continue

                    if "confidence" not in dataframe.columns or "yolo-id" not in dataframe.columns or "unique-id" not in dataframe.columns:  # noqa: E501
                        continue

                    # Keep only rows with confidence >= min_conf
                    dataframe = dataframe.filter(
                        pl.col("confidence").cast(pl.Float64, strict=False) >= float(min_conf)
                    )
                    if dataframe.height == 0:
                        # still record zero rates for non-cellphone metrics if desired; original code skips naturally
                        continue

                    # Helpers to count unique object ids for yolo-id filters
                    def _n_unique_for_yolo_ids(ids: list[int]) -> int:
                        return int(
                            dataframe
                            .filter(pl.col("yolo-id").cast(pl.Int64, strict=False).is_in(ids))
                            .select(pl.col("unique-id").n_unique())
                            .item()
                        )

                    def _n_unique_for_yolo_id(i: int) -> int:
                        return int(
                            dataframe
                            .filter(pl.col("yolo-id").cast(pl.Int64, strict=False) == i)
                            .select(pl.col("unique-id").n_unique())
                            .item()
                        )

                    # ---- CELL PHONES (YOLO 67): per person, normalised ----
                    mobile_ids = _n_unique_for_yolo_id(67)
                    num_person = _n_unique_for_yolo_id(0)
                    if num_person > 0 and mobile_ids > 0 and duration > 0:
                        avg_cellphone = ((mobile_ids * 60) / duration / num_person) * 1000
                        cellphone_info[video_key] = float(avg_cellphone)

                    # ---- TRAFFIC SIGNS (YOLO 9, 11) ----
                    traffic_sign_n = _n_unique_for_yolo_ids([9, 11])
                    traffic_signs_layer[video_key] = float((traffic_sign_n / duration) * 60) if duration > 0 else 0.0

                    # ---- VEHICLES (YOLO 2,3,5,7) ----
                    vehicle_n = _n_unique_for_yolo_ids([2, 3, 5, 7])
                    vehicle_layer[video_key] = float((vehicle_n / duration) * 60) if duration > 0 else 0.0

                    # ---- BICYCLES (YOLO 1) ----
                    bicycle_n = _n_unique_for_yolo_id(1)
                    bicycle_layer[video_key] = float((bicycle_n / duration) * 60) if duration > 0 else 0.0

                    # ---- CARS (YOLO 2) ----
                    car_n = _n_unique_for_yolo_id(2)
                    car_layer[video_key] = float((car_n / duration) * 60) if duration > 0 else 0.0

                    # ---- MOTORCYCLES (YOLO 3) ----
                    motorcycle_n = _n_unique_for_yolo_id(3)
                    motorcycle_layer[video_key] = float((motorcycle_n / duration) * 60) if duration > 0 else 0.0

                    # ---- BUSES (YOLO 5) ----
                    bus_n = _n_unique_for_yolo_id(5)
                    bus_layer[video_key] = float((bus_n / duration) * 60) if duration > 0 else 0.0

                    # ---- TRUCKS (YOLO 7) ----
                    truck_n = _n_unique_for_yolo_id(7)
                    truck_layer[video_key] = float((truck_n / duration) * 60) if duration > 0 else 0.0

                    # ---- PERSONS (YOLO 0) ----
                    person_n = num_person
                    person_layer[video_key] = float((person_n / duration) * 60) if duration > 0 else 0.0

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

        cls._all_metrics_cache = {}

        for i, (metric_name, metric_layer) in enumerate(metric_dicts, 1):
            logger.info(f"[{i}/{len(metric_dicts)}] Wrapping '{metric_name}' ...")
            wrapped = grouping_class.city_country_wrapper(
                input_dict=metric_layer,
                mapping=df_mapping,
                show_progress=True,
            )
            cls._all_metrics_cache[metric_name] = wrapped

    @classmethod
    def _ensure_cache(cls, df_mapping: pl.DataFrame) -> None:
        """
        Ensure that the class-level metrics cache is populated.
        If the cache is empty, computes all metrics for the provided mapping DataFrame.
        """
        if not cls._all_metrics_cache:
            cls._compute_all_metrics(df_mapping)

    @classmethod
    def calculate_cellphones(cls, df_mapping: pl.DataFrame):
        """
        Return the cached cell phone metric, computing all metrics if needed.
        """
        cls._ensure_cache(df_mapping)
        return cls._all_metrics_cache["cellphones"]

    @classmethod
    def calculate_traffic_signs(cls, df_mapping: pl.DataFrame):
        """
        Return the cached traffic sign metric, computing all metrics if needed.
        """
        cls._ensure_cache(df_mapping)
        return cls._all_metrics_cache["traffic_signs"]

    @classmethod
    def calculate_traffic(cls, df_mapping: pl.DataFrame, person: int = 0, bicycle: int = 0, motorcycle: int = 0,
                          car: int = 0, bus: int = 0, truck: int = 0):
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
