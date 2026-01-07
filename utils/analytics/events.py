import os
import common
from tqdm import tqdm
import polars as pl
from custom_logger import CustomLogger
from utils.core.metadata import MetaData

metadata_class = MetaData()
logger = CustomLogger(__name__)  # use custom logger


class Events:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _load_detection_csv(video_key: str) -> pl.DataFrame | None:
        """
        Find and load the detection CSV for `video_key` from configured data folders/subdomains.
        Returns a Polars DataFrame filtered by min_confidence, or None if not found.
        """
        min_conf = float(common.get_configs("min_confidence"))

        for folder_path in common.get_configs("data"):
            for subfolder in common.get_configs("sub_domain"):
                subfolder_path = os.path.join(folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    continue

                for file in os.listdir(subfolder_path):
                    if os.path.splitext(file)[0] == video_key and file.endswith(".csv"):
                        file_path = os.path.join(subfolder_path, file)
                        df = pl.read_csv(file_path)
                        if "confidence" in df.columns:
                            df = df.filter(pl.col("confidence").cast(pl.Float64, strict=False) >= min_conf)
                        return df

        return None

    @staticmethod
    def crossing_event_with_traffic_equipment(df_mapping: pl.DataFrame, data: dict):
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
            # Extract metadata for this video
            result = metadata_class.find_values_with_video_id(df_mapping, video_key)
            if result is None:
                continue

            start_time = result[1]
            end_time = result[2]
            condition = result[3]
            city = result[4]
            latitude = result[6]
            longitude = result[7]
            country = result[8]

            location_key_city = f"{city}_{latitude}_{longitude}_{condition}"
            location_key_country = f"{country}_{condition}"

            # Update total duration per location/condition
            try:
                duration = end_time - start_time
            except Exception:
                continue

            total_duration_by_city[location_key_city] = total_duration_by_city.get(location_key_city, 0) + duration
            total_duration_by_country[location_key_country] = total_duration_by_country.get(location_key_country, 0) + duration  # noqa: E501

            # Load detection CSV
            value = Events._load_detection_csv(video_key)
            if value is None:
                continue

            if not {"unique-id", "frame-count", "yolo-id"}.issubset(set(value.columns)):
                continue

            count_with_equipment = 0
            count_without_equipment = 0

            # Precompute per unique-id: first/last frame-count
            uids = list(crossings.keys())
            if not uids:
                continue

            ranges = (
                value
                .filter(pl.col("unique-id").is_in(uids))
                .group_by("unique-id")
                .agg([
                    pl.col("frame-count").cast(pl.Int64, strict=False).min().alias("_fmin"),
                    pl.col("frame-count").cast(pl.Int64, strict=False).max().alias("_fmax"),
                ])
            )

            # For each crossing id, check if equipment appears within [fmin, fmax]
            for uid, fmin, fmax in ranges.iter_rows():
                seg = value.filter(
                    (pl.col("frame-count").cast(pl.Int64, strict=False) >= int(fmin)) &
                    (pl.col("frame-count").cast(pl.Int64, strict=False) <= int(fmax))
                )
                has_equipment = seg.select(pl.col("yolo-id").is_in([9, 11]).any()).item()
                if bool(has_equipment):
                    count_with_equipment += 1
                else:
                    count_without_equipment += 1

            # Aggregate by city/condition
            crossings_with_traffic_equipment_city[location_key_city] = (
                crossings_with_traffic_equipment_city.get(location_key_city, 0) + count_with_equipment
            )
            crossings_without_traffic_equipment_city[location_key_city] = (
                crossings_without_traffic_equipment_city.get(location_key_city, 0) + count_without_equipment
            )

            # Aggregate by country/condition
            crossings_with_traffic_equipment_country[location_key_country] = (
                crossings_with_traffic_equipment_country.get(location_key_country, 0) + count_with_equipment
            )
            crossings_without_traffic_equipment_country[location_key_country] = (
                crossings_without_traffic_equipment_country.get(location_key_country, 0) + count_without_equipment
            )

        return (
            crossings_with_traffic_equipment_city,
            crossings_without_traffic_equipment_city,
            total_duration_by_city,
            crossings_with_traffic_equipment_country,
            crossings_without_traffic_equipment_country,
            total_duration_by_country,
        )

    # TODO: combine methods for looking at crossing events with/without traffic lights
    @staticmethod
    def crossing_event_wt_traffic_light(df_mapping: pl.DataFrame, data: dict):
        """Plots traffic mortality rate vs percentage of crossing events without traffic light.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
            data (dict): Dictionary containing pedestrian crossing data.
        """
        var_exist, var_nt_exist, ratio = {}, {}, {}
        time_ = []

        counter_1, counter_2 = {}, {}

        for key, df_ids in tqdm(data.items(), total=len(data)):
            counter_exists, counter_nt_exists = 0, 0

            result = metadata_class.find_values_with_video_id(df_mapping, key)
            if result is None:
                continue

            start = result[1]
            end = result[2]
            condition = result[3]
            city = result[4]
            lat = result[6]
            long = result[7]

            duration = end - start
            time_.append(duration)

            value = Events._load_detection_csv(key)
            if value is None:
                continue

            if not {"unique-id", "frame-count", "yolo-id"}.issubset(set(value.columns)):
                continue

            ids = list(df_ids.keys())
            if not ids:
                continue

            ranges = (
                value
                .filter(pl.col("unique-id").is_in(ids))
                .group_by("unique-id")
                .agg([
                    pl.col("frame-count").cast(pl.Int64, strict=False).min().alias("_fmin"),
                    pl.col("frame-count").cast(pl.Int64, strict=False).max().alias("_fmax"),
                ])
            )

            for uid, fmin, fmax in ranges.iter_rows():
                seg = value.filter(
                    (pl.col("frame-count").cast(pl.Int64, strict=False) >= int(fmin)) &
                    (pl.col("frame-count").cast(pl.Int64, strict=False) <= int(fmax))
                )
                yolo_9_exists = seg.select((pl.col("yolo-id") == 9).any()).item()
                if bool(yolo_9_exists):
                    counter_exists += 1
                else:
                    counter_nt_exists += 1

            # Normalise per-minute using last duration (same structure as original)
            if time_[-1] > 0:
                var_exist[key] = ((counter_exists * 60) / time_[-1])
                var_nt_exist[key] = ((counter_nt_exists * 60) / time_[-1])
            else:
                var_exist[key] = 0
                var_nt_exist[key] = 0

            city_id_format = f"{city}_{lat}_{long}_{condition}"
            counter_1[city_id_format] = counter_1.get(city_id_format, 0) + var_exist[key]
            counter_2[city_id_format] = counter_2.get(city_id_format, 0) + var_nt_exist[key]

            denom = counter_1[city_id_format] + counter_2[city_id_format]
            if denom == 0:
                continue

            ratio[city_id_format] = (counter_2[city_id_format] * 100) / denom

        return ratio
