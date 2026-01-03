class Aggregation:
    def __init__(self) -> None:
        pass

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
