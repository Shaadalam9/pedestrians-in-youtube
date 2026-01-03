import common
from utils.core.grouping import Grouping
from utils.core.metadata import MetaData

metadata_class = MetaData()
grouping_class = Grouping()


class Metrics:
    def __init__(self) -> None:
        pass

    def time_to_cross(self, dataframe, ids, video_id, df_mapping):
        """Calculates the time taken for each object with specified IDs to cross the road.

        Args:
            dataframe (DataFrame): The DataFrame (csv file) containing object data.
            ids (list): A list of unique IDs of objects which are crossing the road.

        Returns:
            dict: A dictionary where keys are object IDs and values are the time taken for
            each object to cross the road, in seconds.
        """
        if 'frame-count' not in dataframe.columns:
            return {}

        result = metadata_class.find_values_with_video_id(df_mapping, video_id)

        # Check if the result is None (i.e., no matching data was found)
        if result is not None:
            # Unpack the result since it's not None
            fps = result[17]

        # Initialise an empty dictionary to store time taken for each object to cross
        var = {}

        # Iterate through each object ID
        for id in ids:
            # Find the minimum and maximum x-coordinates for the object's movement
            x_min = dataframe[dataframe["unique-id"] == id]["x-center"].min()
            x_max = dataframe[dataframe["unique-id"] == id]["x-center"].max()

            # Get a sorted group of entries for the current object ID
            sorted_grp = dataframe[dataframe["unique-id"] == id]

            # Get the corresponding frame counts instead of index
            x_min_frame = sorted_grp[sorted_grp['x-center'] == x_min]['frame-count'].iloc[0]
            x_max_frame = sorted_grp[sorted_grp['x-center'] == x_max]['frame-count'].iloc[0]

            time_taken = abs(x_max_frame - x_min_frame) / fps
            var[id] = time_taken

        return var

    def calculate_speed_of_crossing(self, df_mapping, df, data):
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

        # Check if all values in the dict are empty dicts
        if not any(data.values()):
            return None

        time_ = []  # List to store durations of videos (not used in output)
        speed_compelete = {}  # Dictionary to hold valid speed results for each video

        # Create a dictionary to store country information for each city
        city_country_map_ = {}

        # Group YOLO data by unique person ID
        grouped = df.groupby('unique-id')

        # Iterate through all video IDs and their corresponding crossing data
        for key, id_time in data.items():
            speed_id_compelete = {}  # Store valid speeds for individuals in this video

            if id_time == {}:  # Skip if there is no data
                continue

            result = metadata_class.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                # Unpack video metadata (edit if order of unpacked variables changes)
                start = result[1]
                end = result[2]
                city = result[4]
                lat = result[6]
                long = result[7]
                avg_height = result[15]
                iso3 = result[16]

                # Store the country associated with each city
                city_country_map_[f'{city}_{lat}_{long}'] = iso3

                # Calculate total duration of the crossing segment in this video
                duration = end - start
                time_.append(duration)

                # Loop through each individual's crossing data in this video
                for id, time in id_time.items():

                    # Get all frames for this person
                    grouped_with_id = grouped.get_group(id)

                    # Calculate mean height of bounding box for this person
                    mean_height = grouped_with_id['height'].mean()

                    # Find minimum and maximum x-center positions to estimate path length
                    min_x_center = grouped_with_id['x-center'].min()
                    max_x_center = grouped_with_id['x-center'].max()

                    # Estimate "pixels per centi-meter" using average height and actual avg_height
                    ppm = mean_height / avg_height

                    # Estimate real-world distance crossed (in centimeters)
                    distance = (max_x_center - min_x_center) / ppm

                    # Calculate walking speed (meters per second)
                    speed_ = (distance / time) / 100

                    speed_id_compelete[id] = speed_

                # Store all valid speeds for this video
                speed_compelete[key] = speed_id_compelete

        # Group and organise the results for downstream analysis/plotting
        output = grouping_class.city_country_wrapper(input_dict=speed_compelete, mapping=df_mapping)

        return output

    def avg_speed_of_crossing_city(self, df_mapping, all_speed):
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
            city, lat, long, cond = city_lat_lang_condition.split("_")
            box = []
            for video_id, value_2 in value_1.items():
                for unique_id, speed in value_2.items():
                    # Only include the speed if it's within configured bounds
                    if common.get_configs("min_speed_limit") <= speed <= common.get_configs("max_speed_limit"):
                        box.append(speed)
            if len(box) > 0:
                all_speed_city[city_lat_lang_condition] = box
                avg_speed_city[city_lat_lang_condition] = (sum(box) / len(box))

        return avg_speed_city, all_speed_city

    def avg_speed_of_crossing_country(self, df_mapping, all_speed):
        """
        Calculate the average speed for each country based on all_speed data and a mapping DataFrame.

        Args:
            all_speed (dict): Nested dictionary structured as
                {city_lat_lang_condition: {video_id: {unique_id: speed}}}
            df_mapping (pd.DataFrame): DataFrame containing video_id and country information.

        Returns:
            dict: Dictionary mapping each country to its average speed (float).
        """
        avg_speed = {}

        # Iterate through each city condition in the all_speed dict
        for city_lat_lang_condition, value_1 in all_speed.items():
            # For each video_id, retrieve speeds
            for video_id, value_2 in value_1.items():
                # Find the country associated with the current video_id
                result = metadata_class.find_values_with_video_id(df=df_mapping, key=video_id)
                if result is not None:
                    condition = result[3]
                    country = result[8]

                    for unique_id, speed in value_2.items():
                        # Only include the speed if it's within configured bounds
                        if common.get_configs("min_speed_limit") <= speed <= common.get_configs("max_speed_limit"):
                            if f'{country}_{condition}' not in avg_speed:
                                avg_speed[f'{country}_{condition}'] = []
                            avg_speed[f'{country}_{condition}'].append(speed)

        # Now, calculate the average speed for each country
        avg_speed_result = {}
        for country_condition, speed_list in avg_speed.items():
            if speed_list:  # Avoid division by zero
                avg_speed_result[country_condition] = sum(speed_list) / len(speed_list)

        return avg_speed_result, avg_speed

    def time_to_start_cross(self, df_mapping, df, data, person_id=0):
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
        # Check if all values in the dict are empty dicts
        if not any(data.values()):
            return None

        time_compelete = {}

        data_cross = {}
        time_id_complete = {}

        # Group YOLO data by unique person ID
        crossed_ids_grouped = df.groupby('unique-id')

        # Extract relevant information using the find_values function
        result = metadata_class.find_values_with_video_id(df_mapping, next(iter(data)))

        # Check if the result is None (i.e., no matching data was found)
        if result is not None:
            fps = result[17]

            checks_per_second = common.get_configs("check_per_sec_time")
            interval_seconds = 1 / checks_per_second  # 0.333...
            step = max(1, int(round(interval_seconds * fps)))  # Frames between checks (at least 1)

            # Directly get the inner dictionary
            inner_dict = next(iter(data.values()))

            for unique_id, time in inner_dict.items():
                group_data = crossed_ids_grouped.get_group(unique_id)
                x_values = group_data["x-center"].values
                initial_x = x_values[0]  # Initial x-value
                mean_height = group_data['height'].mean()
                flag = 0
                margin = 0.1 * mean_height  # Margin for considering crossing event
                consecutive_frame = 0

                stop = len(x_values) - step

                for i in range(0, stop, step):
                    # Indexing is safe because step is int
                    current_x = x_values[i]
                    next_x = x_values[i + step]

                    if initial_x < 0.5:  # Check if crossing from left to right
                        if (current_x - margin <= next_x <= current_x + margin):
                            consecutive_frame += 1
                            if consecutive_frame == 3:  # Check for three consecutive frames
                                flag = 1
                        elif flag == 1:
                            data_cross[unique_id] = consecutive_frame
                            break
                        else:
                            consecutive_frame = 0

                    else:  # Check if crossing from right to left
                        if (current_x - margin >= next_x >= current_x + margin):
                            consecutive_frame += 1
                            if consecutive_frame == 3:  # Check for three consecutive frames
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

            time_compelete[next(iter(data))] = time_id_complete

        output = grouping_class.city_country_wrapper(input_dict=time_compelete, mapping=df_mapping)

        return output

    def avg_time_to_start_cross_city(self, df_mapping, all_time):
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

        # Iterate over each city condition in the all_time dictionary
        for city_condition, value_1 in all_time.items():
            box = []  # List to store adjusted times for the current city condition

            # Iterate over each video_id and its times
            for video_id, value_2 in value_1.items():
                if value_2 is None:
                    continue  # Skip if there are no times for this video
                else:
                    # Retrieve fps value from the mapping using video_id
                    result = metadata_class.find_values_with_video_id(df_mapping, video_id)
                    if result is not None:

                        # Adjust time for each unique_id if it is positive
                        for unique_id, time in value_2.items():
                            if time > 0:
                                time_in_seconds = time / common.get_configs("check_per_sec_time")

                                # https://doi.org/10.1016/j.jtte.2015.12.001
                                if common.get_configs("min_waiting_time") <= time_in_seconds <= common.get_configs("max_waiting_time"):  # noqa: E501
                                    box.append(time_in_seconds)

            # Compute average adjusted time for the current city condition
            if len(box) > 0:
                all_time_city[city_condition] = box
                avg_time_city[city_condition] = sum(box) / len(box)

        return avg_time_city, all_time_city

    def avg_time_to_start_cross_country(self, df_mapping, all_time):
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
        avg_over_time = {}

        # Iterate over all city conditions in the all_time dictionary
        for city_condition, videos in all_time.items():
            # For each video_id and its times
            for video_id, times in videos.items():
                if times is None:
                    continue  # Skip if no times for this video

                # Retrieve mapping info using video_id
                result = metadata_class.find_values_with_video_id(df_mapping, video_id)
                if result is not None:
                    condition = result[3]
                    country = result[8]

                    # Adjust and store each valid time for the current country
                    for unique_id, time in times.items():
                        if time > 0:
                            # Convert in real world second from the number of detection
                            time_in_seconds = time / common.get_configs("check_per_sec_time")
                            if common.get_configs("min_waiting_time") <= time_in_seconds <= common.get_configs("max_waiting_time"):  # noqa: E501
                                if f'{country}_{condition}' not in avg_over_time:
                                    avg_over_time[f'{country}_{condition}'] = []
                                avg_over_time[f'{country}_{condition}'].append(time_in_seconds)

        # Compute the average adjusted time per country
        avg_over_time_result = {}
        for country_condition, time_list in avg_over_time.items():
            if time_list:  # Avoid division by zero
                avg_over_time_result[country_condition] = sum(time_list) / len(time_list)

        return avg_over_time_result, avg_over_time
