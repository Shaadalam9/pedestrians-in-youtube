import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
import ast
import pandas as pd
import math

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class Values():
    def __init__(self) -> None:
        pass

    def find_values_with_video_id(self, df, key):
        """Extracts relevant data from a DataFrame based on a given key.

        Args:
            df (DataFrame): The DataFrame containing the data.
            key (str): The key to search for in the DataFrame.

        Returns:
            tuple: A tuple containing information related to the key, including:
                - Video ID
                - Start time
                - End time
                - Time of day
                - City
                - State
                - Latitude
                - Longitude
                - Country
                - GDP per capita
                - Population
                - Population of the country
                - Traffic mortality
                - Continent
                - Literacy rate
                - Average height
                - ISO-3 code for country
                - Fps of the video
                - Type of vehicle
        """
        id, start_, fps = key.rsplit("_", 2)  # Splitting the key into video ID and start time

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Extracting data from the DataFrame row
            video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])
            time_of_day = ast.literal_eval(row["time_of_day"])
            city = row["city"]
            state = row['state'] if not pd.isna(row['state']) else "unknown"
            latitude = row["lat"]
            longitude = row["lon"]
            country = row["country"]
            gdp = row["gmp"]
            population = row["population_city"]
            population_country = row["population_country"]
            traffic_mortality = row["traffic_mortality"]
            continent = row["continent"]
            literacy_rate = row["literacy_rate"]
            avg_height = row["avg_height"]
            iso3 = row["iso3"]
            vehicle_type = ast.literal_eval(row["vehicle_type"])

            # Iterate through each video, start time, end time, and time of day
            for video, start, end, time_of_day_, vehicle_type, in zip(video_ids, start_times, end_times, time_of_day, vehicle_type):  # noqa: E501
                # Check if the current video matches the specified ID
                if video == id:
                    logger.debug(f"Finding values for {video} start={start}, end={end}")
                    counter = 0
                    # Iterate through each start time
                    for s in start:
                        # Check if the start time matches the specified start time
                        if int(start_) == s:
                            # Calculate gpd per capita to avoid division by zero
                            if int(population) > 0:
                                gpd_capita = int(gdp)/int(population)
                            else:
                                gpd_capita = 0
                            # Return relevant information once found
                            return (video,                      # 0
                                    s,                          # 1
                                    end[counter],               # 2
                                    time_of_day_[counter],      # 3
                                    city,                       # 4
                                    state,                      # 5
                                    latitude,                   # 6
                                    longitude,                  # 7
                                    country,                    # 8
                                    gpd_capita,                 # 9
                                    population,                 # 10
                                    population_country,         # 11
                                    traffic_mortality,          # 12
                                    continent,                  # 13
                                    literacy_rate,              # 14
                                    avg_height,                 # 15
                                    iso3,                       # 16
                                    int(fps),                   # 17
                                    vehicle_type)               # 18
                        counter += 1

    def get_value(self, df, column_name1, column_value1, column_name2, column_value2, target_column):
        """
        Retrieves a value from the target_column based on the condition
        that both column_name1 matches column_value1 and column_name2 matches column_value2.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the mapping file.
        column_name1 (str): The first column to search for the matching value.
        column_value1 (str): The value to search for in column_name1.
        column_name2 (str): The second column to search for the matching value.
        column_value2 (str): The value to search for in column_name2. If "unknown", the value is treated as NaN.
        target_column (str): The column from which to retrieve the corresponding value.

        Returns:
        Any: The value from target_column that corresponds to the matching values in both
             column_name1 and column_name2.
        """
        if column_name2 is None or column_value2 is None:
            result = df[df[column_name1] == column_value1][target_column]
            if not result.empty:
                return result.iloc[0]
            else:
                return None

        else:
            # Treat column_value2 as NaN if it is "unknown"
            if column_value2 == "unknown":
                column_value2 = float('nan')

            # Filter the DataFrame where both conditions are met
            if pd.isna(column_value2):
                result = df[(df[column_name1] == column_value1) & (df[column_name2].isna())][target_column]
            else:
                result = df[(df[column_name1] == column_value1) & (df[column_name2] == column_value2)][target_column]

            # Check if the result is not empty (i.e., if there is a match)
            if not result.empty:
                # Return the first matched value
                return result.values[0]
            else:
                # Return None if no matching value is found
                return None

    def calculate_total_seconds_for_city(self, df, city_name, state_name):
        """Calculates the total number of seconds of video for a given city and state.

        This method searches a DataFrame for a specific city and state, retrieves
        start and end time lists from matching rows, and sums the total recorded
        video time in seconds. It supports cases where the state is `"unknown"`
        (matching rows with missing state values).

        Args:
            df (pd.DataFrame): DataFrame containing city, state, start_time, and end_time columns.
                - `start_time` and `end_time` should be stored as string representations of nested lists.
            city_name (str): Name of the city to match.
            state_name (str): Name of the state to match, or `"unknown"` for missing state.

        Returns:
            int: Total video duration in seconds for the given city and state.

        Examples:
            >>> df = pd.DataFrame({
            ...     'city': ['Paris'],
            ...     'state': [None],
            ...     'start_time': ["[[0, 10], [20, 30]]"],
            ...     'end_time': ["[[5, 15], [25, 35]]"]
            ... })
            >>> calculate_total_seconds_for_city(df, "Paris", "unknown")
            10
        """
        # Filter the DataFrame for the specific city and state
        if state_name.lower() == "unknown":
            row = df[(df["city"] == city_name) & (pd.isna(df["state"]))]
        else:
            row = df[(df["city"] == city_name) & (df["state"] == state_name)]

        # Return 0 if no match is found
        if row.empty:
            return 0  # Could alternatively raise an exception

        # Extract the first matching row
        row = row.iloc[0]

        # Convert stored string representations of lists into Python lists
        start_times = ast.literal_eval(row["start_time"])
        end_times = ast.literal_eval(row["end_time"])

        total_seconds = 0

        # Iterate through nested start/end times and accumulate durations
        for start, end in zip(start_times, end_times):
            for s, e in zip(start, end):
                total_seconds += (int(e) - int(s))

        return total_seconds

    def remove_columns_below_threshold(self, df, threshold):
        """Removes `start_time`/`end_time` column pairs where total recorded time is below a threshold.

        This method scans all columns in the DataFrame for matching `start_timeX` /
        `end_timeX` column pairs (where X can be a suffix), calculates the total
        video duration for all rows in each pair, and removes pairs whose total
        duration is less than the given threshold.

        Args:
            df (pd.DataFrame): DataFrame containing video start/end time columns.
                - Start/end times should be stored as string representations of nested lists.
            threshold (int): Minimum total seconds required for a start/end column
                pair to be retained.

        Returns:
            pd.DataFrame: A modified DataFrame with low-duration column pairs removed.

        Example:
            >>> df = pd.DataFrame({
            ...     'start_time_city1': ["[[0, 10]]", "[[20, 30]]"],
            ...     'end_time_city1': ["[[5, 15]]", "[[25, 35]]"]
            ... })
            >>> remove_columns_below_threshold(df, threshold=15)
            Empty DataFrame
            Columns: []
            Index: [0, 1]
        """
        cols_to_remove = []

        # Iterate over all columns to find start/end pairs
        for col in df.columns:
            if col.startswith("start_time") or col.startswith("end_time"):
                # Identify the base name to find its matching pair
                base = col.replace("start_time", "").replace("end_time", "")
                start_col = f"start_time{base}"
                end_col = f"end_time{base}"

                # Only process if both start and end columns exist
                if start_col in df.columns and end_col in df.columns:
                    total_seconds = 0

                    # Iterate through each row and accumulate durations
                    for _, row in df.iterrows():
                        start_times = ast.literal_eval(row[start_col])
                        end_times = ast.literal_eval(row[end_col])

                        # Sum the duration of each start/end pair
                        for start, end in zip(start_times, end_times):
                            for s, e in zip(start, end):
                                total_seconds += (int(e) - int(s))

                    # Mark this start/end pair for removal if below threshold
                    if total_seconds < threshold:
                        cols_to_remove += [start_col, end_col]

        # Deduplicate the removal list and drop columns from DataFrame
        cols_to_remove = list(set(cols_to_remove))
        df_modified = df.drop(columns=cols_to_remove)

        return df_modified

    def calculate_total_seconds(self, df):
        """Calculates the total video duration (in seconds) from a mapping DataFrame.

        This method reads `start_time` and `end_time` columns, which contain string
        representations of nested lists of timestamps. It iterates through all rows,
        summing the total video duration across the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing at least `start_time` and `end_time` columns.

        Returns:
            int: The total number of seconds across all start/end pairs in the DataFrame.

        Example:
            >>> df = pd.DataFrame({
            ...     'start_time': ["[[0, 10]]", "[[20, 30]]"],
            ...     'end_time': ["[[5, 15]]", "[[25, 35]]"]
            ... })
            >>> calculate_total_seconds(df)
            20
        """
        grand_total_seconds = 0

        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            # Convert string representation into Python list
            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])

            # Sum duration for each start/end timestamp pair
            for start, end in zip(start_times, end_times):
                for s, e in zip(start, end):
                    grand_total_seconds += (int(e) - int(s))

        return grand_total_seconds

    def calculate_total_videos(self, df):
        """Counts the total number of unique videos from a mapping DataFrame.

        This method reads the `videos` column, splits its comma-separated contents,
        strips extra whitespace, and counts unique video IDs/names.

        Args:
            df (pd.DataFrame): DataFrame containing a `videos` column with
                comma-separated video identifiers.

        Returns:
            int: The total count of unique videos.

        Example:
            >>> df = pd.DataFrame({
            ...     'videos': ["video1, video2", "video2, video3"]
            ... })
            >>> calculate_total_videos(df)
            3
        """
        total_videos = set()

        # Iterate through each row and add all videos to a set (for uniqueness)
        for _, row in df.iterrows():
            videos_list = row["videos"].split(",")  # Split comma-separated values
            for video in videos_list:
                total_videos.add(video.strip())  # Remove whitespace before adding

        return len(total_videos)

    def pedestrian_cross_per_city(self, pedestrian_crossing_count, df_mapping):
        """Aggregates pedestrian crossing counts per city-condition key.

        This method:
            1. Counts pedestrian crossing events per video ID from the input dictionary.
            2. Uses the mapping DataFrame to look up each video's city, latitude, longitude, and condition.
            3. Constructs a unique key in the format: `{city}_{lat}_{long}_{condition}`.
            4. Aggregates counts for each unique city-condition key.

        Args:
            pedestrian_crossing_count (dict): Dictionary mapping video IDs to dictionaries
                containing:
                    - "ids" (list): List of detected pedestrian crossing event IDs.
            df_mapping (pd.DataFrame): Mapping DataFrame used to look up video metadata.
                Expected to be compatible with `values_class.find_values_with_video_id`.

        Returns:
            dict: Dictionary mapping `{city}_{lat}_{long}_{condition}` → total pedestrian crossing count.

        Example:
            >>> pedestrian_crossing_count = {
            ...     "vid123": {"ids": [1, 2, 3]},
            ...     "vid456": {"ids": [4]}
            ... }
            >>> # df_mapping must work with values_class.find_values_with_video_id
            >>> pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping)
            {
                "Paris_48.85_2.35_0": 3,
                "London_51.51_-0.13_1": 1
            }
        """
        final = {}

        # Step 1: Count pedestrian crossing events per video
        count = {key: len(value["ids"]) for key, value in pedestrian_crossing_count.items()}

        # Step 2: Map each video to its city and condition
        for key, total_events in count.items():
            result = self.find_values_with_video_id(df_mapping, key)

            if result is not None:
                condition = result[3]
                city = result[4]
                lat = result[6]
                long = result[7]

                # Step 3: Create the unique city-condition key
                city_time_key = f"{city}_{lat}_{long}_{condition}"

                # Step 4: Aggregate counts for this key
                if city_time_key in final:
                    final[city_time_key] += total_events
                else:
                    final[city_time_key] = total_events

        return final

    def pedestrian_cross_per_country(self, pedestrian_cross_city, df_mapping):
        """Aggregates pedestrian crossing counts from city level to country level.

        This method converts city-level pedestrian crossing counts into
        country-level totals, grouped by country and condition.

        Args:
            pedestrian_cross_city (dict): Dictionary mapping
                `{city}_{lat}_{long}_{condition}` → pedestrian crossing count (int).
            df_mapping (pd.DataFrame): Mapping DataFrame containing city-to-country
                information. Must be compatible with `values_class.get_value`.

        Returns:
            dict: Dictionary mapping `{country}_{condition}` → total pedestrian crossing count.

        Example:
            >>> pedestrian_cross_city = {
            ...     "Paris_48.85_2.35_0": 3,
            ...     "London_51.51_-0.13_1": 2
            ... }
            >>> # df_mapping must work with values_class.get_value
            >>> pedestrian_cross_per_country(pedestrian_cross_city, df_mapping)
            {
                "France_0": 3,
                "United Kingdom_1": 2
            }
        """
        final = {}

        # Iterate through each city-condition entry
        for city_lat_long_cond, value in pedestrian_cross_city.items():
            # Extract city, latitude, and condition
            city, lat, _, cond = city_lat_long_cond.split("_")

            # Look up the country for this city-lat combination
            country = self.get_value(df_mapping, "city", city, "lat", float(lat), "country")

            # Aggregate counts by country-condition
            country_key = f"{country}_{cond}"
            if country_key in final:
                final[country_key] += value
            else:
                final[country_key] = value

        return final

    def safe_average(self, values):
        """Calculates the average of a list, ignoring None and NaN values.

        This method removes any `None` or `NaN` entries from the list before
        calculating the mean. If no valid values remain, it returns `0`.

        Args:
            values (list): List of numeric values, possibly containing `None` or `NaN`.

        Returns:
            float: The average of valid values, or `0` if none are valid.

        Example:
            >>> safe_average([1, 2, None, float('nan'), 4])
            2.3333333333333335
            >>> safe_average([None, float('nan')])
            0
        """
        # Keep only values that are not None and not NaN
        valid_values = [
            v for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]

        # Return average if list is not empty; otherwise return 0
        return sum(valid_values) / len(valid_values) if valid_values else 0
