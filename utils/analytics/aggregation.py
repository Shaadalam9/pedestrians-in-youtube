import polars as pl


class Aggregation:
    def __init__(self) -> None:
        pass

    @staticmethod
    def aggregate_by_iso3(df: pl.DataFrame) -> pl.DataFrame:
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

        df2 = df.drop(drop_columns)

        # --- Continent majority (mode-like) ---
        continent_majority = None
        if "continent" in df2.columns:
            continent_majority = (
                df2
                .group_by(["iso3", "continent"])
                .len()
                .sort(["iso3", "len"], descending=[False, True])
                .group_by("iso3")
                .agg(pl.col("continent").first().alias("continent"))
            )
            # avoid aggregating continent twice
            static_columns_no_cont = [c for c in static_columns if c != "continent"]
            df2_no_cont = df2.drop("continent")
        else:
            static_columns_no_cont = static_columns
            df2_no_cont = df2

        agg_exprs: list[pl.Expr] = []

        # "first" for static columns
        for c in static_columns_no_cont:
            agg_exprs.append(pl.col(c).first().alias(c))

        # collect into list for merge columns (works for Utf8; avoids List-only ops)
        for c in merge_columns:
            agg_exprs.append(pl.col(c).implode().alias(c))

        # sum for numeric columns (cast safely)
        for c in sum_columns:
            agg_exprs.append(
                pl.col(c)
                .cast(pl.Float64, strict=False)
                .fill_null(0)
                .sum()
                .alias(c)
            )

        df_grouped = df2_no_cont.group_by("iso3").agg(agg_exprs)

        if continent_majority is not None:
            df_grouped = df_grouped.join(continent_majority, on="iso3", how="left")

        return df_grouped
