import polars as pl
import common

from utils.core.grouping import Grouping
from utils.core.metadata import MetaData

metadata_class = MetaData()
grouping_class = Grouping()


class Metrics:
    def __init__(self) -> None:
        pass

    def time_to_cross(self, dataframe: pl.DataFrame, ids: list, video_id: str, df_mapping: pl.DataFrame) -> dict:
        """Calculates the time taken for each object with specified IDs to cross the road.

        Args:
            dataframe (DataFrame): The DataFrame (csv file) containing object data.
            ids (list): A list of unique IDs of objects which are crossing the road.

        Returns:
            dict: A dictionary where keys are object IDs and values are the time taken for
            each object to cross the road, in seconds.
        """
        required = {"frame-count", "unique-id", "x-center"}
        if not required.issubset(set(dataframe.columns)):
            return {}
        if not ids:
            return {}

        result = metadata_class.find_values_with_video_id(df_mapping, video_id)
        if result is None:
            return {}

        fps = result[17]
        try:
            fps = float(fps)
        except Exception:
            return {}

        if fps <= 0:
            return {}

        df_ids = dataframe.filter(pl.col("unique-id").is_in(ids))

        # Per unique-id: find frame at min/max x-center, then time = |frame_max - frame_min| / fps
        agg = (
            df_ids
            .group_by("unique-id")
            .agg([
                pl.col("x-center").cast(pl.Float64, strict=False).min().alias("_x_min"),
                pl.col("x-center").cast(pl.Float64, strict=False).max().alias("_x_max"),
                pl.col("frame-count")
                  .cast(pl.Int64, strict=False)
                  .filter(pl.col("x-center").cast(pl.Float64, strict=False) == pl.col("x-center").cast(pl.Float64, strict=False).min())  # noqa: E501
                  .first()
                  .alias("_x_min_frame"),
                pl.col("frame-count")
                  .cast(pl.Int64, strict=False)
                  .filter(pl.col("x-center").cast(pl.Float64, strict=False) == pl.col("x-center").cast(pl.Float64, strict=False).max())  # noqa: E501
                  .first()
                  .alias("_x_max_frame"),
            ])
            .with_columns(
                ((pl.col("_x_max_frame") - pl.col("_x_min_frame")).abs() / pl.lit(fps)).alias("_time_taken")
            )
            .select(["unique-id", "_time_taken"])
        )

        var: dict = {}
        for uid, t in agg.iter_rows():
            try:
                var[uid] = float(t)
            except Exception:
                continue
        return var

    def calculate_speed_of_crossing(self, df_mapping: pl.DataFrame, df: pl.DataFrame, data: dict):
        """
        Calculate and organise the walking speeds of individuals crossing in various videos,
        grouping the results by city, state, and crossing condition.

        Args:
            df_mapping (pd.DataFrame): DataFrame mapping video IDs to metadata including
                city, state, country, and other contextual details.
            df (dict): Dictionary containing DataFrames extracted from YOLO for each video (keyed by video ID).
            data (dict): Dictionary where keys are video IDs and values are dictionaries
                mapping person IDs to crossing durations (in frames or seconds).

        Returns:
            dict: A dictionary where each key is a combination of 'city_state_condition'
                mapping to a list of walking speeds (in m/s) for valid crossings.
        """

        if not any(data.values()):
            return None

        required = {"unique-id", "x-center", "height"}
        if not required.issubset(set(df.columns)):
            return None

        speed_complete: dict[str, dict] = {}

        for key, id_time in data.items():
            if not id_time:
                continue

            result = metadata_class.find_values_with_video_id(df_mapping, key)
            if result is None:
                continue

            avg_height = result[15]
            try:
                avg_height = float(avg_height)  # pyright: ignore[reportArgumentType]
            except Exception:
                continue
            if avg_height <= 0:
                continue

            # Build a small table of crossing times
            ids = list(id_time.keys())
            times = list(id_time.values())
            times_df = pl.DataFrame({"unique-id": ids, "_cross_time": times})

            # Aggregate per person id from detections
            stats = (
                df.filter(pl.col("unique-id").is_in(ids))
                .group_by("unique-id")
                .agg([
                      pl.col("height").cast(pl.Float64, strict=False).mean().alias("_mean_height"),
                      pl.col("x-center").cast(pl.Float64, strict=False).min().alias("_min_x"),
                      pl.col("x-center").cast(pl.Float64, strict=False).max().alias("_max_x"),
                  ])
            )

            joined = stats.join(times_df, on="unique-id", how="inner")

            # ppm = mean_height / avg_height
            # distance_cm = (max_x - min_x) / ppm
            # speed_mps = (distance_cm / time) / 100
            speeds = (
                joined
                .with_columns([
                    (pl.col("_mean_height") / pl.lit(avg_height)).alias("_ppm"),
                ])
                .with_columns([
                    pl.when((pl.col("_cross_time") > 0) & (pl.col("_ppm") > 0))
                      .then(((pl.col("_max_x") - pl.col("_min_x")) / pl.col("_ppm") / pl.col("_cross_time")) / 100.0)
                      .otherwise(None)
                      .alias("_speed_mps")
                ])
                .drop_nulls(["_speed_mps"])
                .select(["unique-id", "_speed_mps"])
            )

            speed_id_complete: dict = {}
            for uid, sp in speeds.iter_rows():
                try:
                    speed_id_complete[uid] = float(sp)
                except Exception:
                    continue

            speed_complete[key] = speed_id_complete

        if not speed_complete:
            return None

        output = grouping_class.city_country_wrapper(input_dict=speed_complete, mapping=df_mapping)
        return output

    def avg_speed_of_crossing_city(self, df_mapping: pl.DataFrame, all_speed: dict):
        """
        Calculate the average crossing speed for each city-condition combination.

        This function uses `calculate_speed_of_crossing` to obtain a nested dictionary of speed values,
        flattens the structure, and computes the average speed for each `city_condition`.

        Args:
            df_mapping (pd.DataFrame): Mapping DataFrame with city metadata.
            dfs (dict): Dictionary of DataFrames for each video segment.
            data (dict): Input data used to compute crossing speeds.

        Returns:
            dict: A dictionary where keys are city-condition strings and values are average speeds.
        """
        avg_speed_city, all_speed_city = {}, {}

        for city_lat_lang_condition, value_1 in all_speed.items():
            box = []
            for _video_id, value_2 in value_1.items():
                for _unique_id, speed in value_2.items():
                    if common.get_configs("min_speed_limit") <= speed <= common.get_configs("max_speed_limit"):
                        box.append(speed)
            if box:
                all_speed_city[city_lat_lang_condition] = box
                avg_speed_city[city_lat_lang_condition] = sum(box) / len(box)

        return avg_speed_city, all_speed_city

    def avg_speed_of_crossing_country(self, df_mapping: pl.DataFrame, all_speed: dict):
        """
        Calculate the average speed for each country based on all_speed data and a mapping DataFrame.

        Args:
            all_speed (dict): Nested dictionary structured as
                {city_lat_lang_condition: {video_id: {unique_id: speed}}}
            df_mapping (pd.DataFrame): DataFrame containing video_id and country information.

        Returns:
            dict: Dictionary mapping each country to its average speed (float).
        """
        avg_speed: dict[str, list[float]] = {}

        for _city_lat_lang_condition, value_1 in all_speed.items():
            for video_id, value_2 in value_1.items():
                result = metadata_class.find_values_with_video_id(df=df_mapping, key=video_id)
                if result is None:
                    continue
                condition = result[3]
                country = result[8]

                for _unique_id, speed in value_2.items():
                    if common.get_configs("min_speed_limit") <= speed <= common.get_configs("max_speed_limit"):
                        k = f"{country}_{condition}"
                        avg_speed.setdefault(k, []).append(speed)

        avg_speed_result = {k: (sum(v) / len(v)) for k, v in avg_speed.items() if v}
        return avg_speed_result, avg_speed

    def time_to_start_cross(self, df_mapping: pl.DataFrame, df: pl.DataFrame, data: dict, person_id: int = 0):
        """
        Calculate the time to start crossing the road of individuals crossing in various videos
        and organise them by city, state, and condition.

        Args:
            df_mapping (dataframe): A DataFrame mapping video IDs to metadata such as
                city, state, country, and other contextual information.
            df (dict): A dictionary where contains all the csv files extracted from YOLO.
            data (dict): A dictionary where keys are video IDs and values are dictionaries
                mapping person IDs to crossing durations.
            person_id (int, optional): YOLO unique representation for person

        Returns:
            speed_dict (dict): A dictionary with keys formatted as 'city_state_condition' mapping to lists
                of walking speeds (m/s) for each valid crossing.
            all_speed (list): A flat list of all calculated walking speeds (m/s) across videos, including outliers.
        """
        if not any(data.values()):
            return None

        required = {"unique-id", "frame-count", "x-center", "height"}
        if not required.issubset(set(df.columns)):
            return None

        key0 = next(iter(data))
        result = metadata_class.find_values_with_video_id(df_mapping, key0)
        if result is None:
            return None

        fps = result[17]
        try:
            fps = float(fps)
        except Exception:
            return None
        if fps <= 0:
            return None

        checks_per_second = common.get_configs("check_per_sec_time")
        interval_seconds = 1 / checks_per_second
        step = max(1, int(round(interval_seconds * fps)))

        inner_dict = next(iter(data.values()))  # {unique_id: time}
        time_id_complete: dict = {}
        data_cross: dict = {}

        for unique_id, _time in inner_dict.items():
            group_data = (
                df.filter(pl.col("unique-id") == unique_id)
                  .sort("frame-count")
                  .select(["x-center", "height"])
            )

            if group_data.height == 0:
                continue

            x_values = group_data.get_column("x-center").to_numpy()
            try:
                mean_height = float(group_data.select(pl.col("height").cast(pl.Float64, strict=False).mean()).item())
            except Exception:
                continue

            if x_values.size == 0:
                continue

            initial_x = x_values[0]
            flag = 0
            margin = 0.1 * mean_height
            consecutive_frame = 0

            stop = int(x_values.size) - step
            if stop <= 0:
                continue

            for i in range(0, stop, step):
                current_x = x_values[i]
                next_x = x_values[i + step]

                if initial_x < 0.5:  # left -> right
                    if (current_x - margin <= next_x <= current_x + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            flag = 1
                    elif flag == 1:
                        data_cross[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0
                else:  # right -> left (kept as-is from original logic)
                    if (current_x - margin >= next_x >= current_x + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            flag = 1
                    elif flag == 1:
                        data_cross[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

            if consecutive_frame >= 3:
                time_id_complete[unique_id] = consecutive_frame

        if len(data_cross) == 0:
            return None

        time_complete = {key0: time_id_complete}
        output = grouping_class.city_country_wrapper(input_dict=time_complete, mapping=df_mapping)
        return output

    def avg_time_to_start_cross_city(self, df_mapping: pl.DataFrame, all_time: dict):
        """
        Calculate the average adjusted time to start crossing for each city condition.

        The time for each entry is adjusted by dividing by (fps / 10), where fps is
        extracted from the mapping DataFrame for the corresponding video_id.

        Args:
            df_mapping (pd.DataFrame): DataFrame containing video_id and fps information.
            all_time (dict): Nested dictionary structured as
                {city_condition: {video_id: {unique_id: time}}}

        Returns:
            dict: Dictionary mapping each city_condition to its average adjusted crossing time (float).
        """
        avg_time_city, all_time_city = {}, {}

        for city_condition, value_1 in all_time.items():
            box = []
            for video_id, value_2 in value_1.items():
                if value_2 is None:
                    continue

                for _unique_id, t in value_2.items():
                    if t > 0:
                        time_in_seconds = t / common.get_configs("check_per_sec_time")
                        if common.get_configs("min_waiting_time") <= time_in_seconds <= common.get_configs("max_waiting_time"):  # noqa: E501
                            box.append(time_in_seconds)

            if box:
                all_time_city[city_condition] = box
                avg_time_city[city_condition] = sum(box) / len(box)

        return avg_time_city, all_time_city

    def avg_time_to_start_cross_country(self, df_mapping: pl.DataFrame, all_time: dict):
        """
        Calculate the average adjusted time to start crossing for each country.

        The time for each entry is adjusted by dividing by (fps / 10), where fps is
        extracted from the mapping DataFrame for the corresponding video_id.

        Args:
            df_mapping (pd.DataFrame): DataFrame containing video_id, fps, and country information.
            all_time (dict): Nested dictionary structured as
                {city_condition: {video_id: {unique_id: time}}}

        Returns:
            dict: Dictionary mapping each country to its average adjusted crossing time (float).
        """
        avg_over_time: dict[str, list[float]] = {}

        for _city_condition, videos in all_time.items():
            for video_id, times in videos.items():
                if times is None:
                    continue

                result = metadata_class.find_values_with_video_id(df_mapping, video_id)
                if result is None:
                    continue
                condition = result[3]
                country = result[8]

                for _unique_id, t in times.items():
                    if t > 0:
                        time_in_seconds = t / common.get_configs("check_per_sec_time")
                        if common.get_configs("min_waiting_time") <= time_in_seconds <= common.get_configs("max_waiting_time"):  # noqa: E501
                            k = f"{country}_{condition}"
                            avg_over_time.setdefault(k, []).append(time_in_seconds)

        avg_over_time_result = {k: (sum(v) / len(v)) for k, v in avg_over_time.items() if v}
        return avg_over_time_result, avg_over_time
