# by Shadab Alam <md_shadab_alam@outlook.com>
import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
from utils.values import Values
from utils.wrappers import Wrappers
import numpy as np
from helper_script import Youtube_Helper

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

values_class = Values()
wrapper_class = Wrappers()
helper = Youtube_Helper()


class Algorithms():
    def __init__(self) -> None:
        pass

    def pedestrian_crossing(self, dataframe, video_id, df_mapping, min_x, max_x, person_id):
        """Counts the number of person with a specific ID crosses the road within specified boundaries.

        Args:
            dataframe (DataFrame): DataFrame containing data from the video.
            min_x (float): Min/Max x-coordinate boundary for the road crossing.
            max_x (float): Max/Min x-coordinate boundary for the road crossing.
            person_id (int): Unique ID assigned by the YOLO tracker to identify the person.

        Returns:
            Tuple[int, list]: A tuple containing the number of person crossed the road within
            the boundaries and a list of unique IDs of the person.
        """

        # Filter dataframe to include only entries for the specified person
        crossed_ids = dataframe[(dataframe["yolo-id"] == person_id)]

        # Group entries by Unique ID
        crossed_ids_grouped = crossed_ids.groupby("unique-id")

        # Filter entries based on x-coordinate boundaries
        filtered_crossed_ids = crossed_ids_grouped.filter(
            lambda x: (x["x-center"] <= min_x).any() and (x["x-center"] >= max_x).any())

        # Get unique IDs of the person who crossed the road within boundaries
        crossed_ids = filtered_crossed_ids["unique-id"].unique()

        result = values_class.find_values_with_video_id(df_mapping, video_id)
        if result is not None:
            avg_height = result[15]

        pedestrian_ids = []
        for uid in crossed_ids:
            if self.is_rider_id(dataframe, uid, avg_height):
                continue  # Filter out riders

            if not self.is_valid_crossing(dataframe, uid):
                continue  # Skip fake crossing

            pedestrian_ids.append(uid)

        return pedestrian_ids, crossed_ids

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

        result = values_class.find_values_with_video_id(df_mapping, video_id)

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

            result = values_class.find_values_with_video_id(df_mapping, key)

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
        output = wrapper_class.city_country_wrapper(input_dict=speed_compelete, mapping=df_mapping)

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
                result = values_class.find_values_with_video_id(df=df_mapping, key=video_id)
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
        result = values_class.find_values_with_video_id(df_mapping, next(iter(data)))

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

        output = wrapper_class.city_country_wrapper(input_dict=time_compelete, mapping=df_mapping)

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
                    result = values_class.find_values_with_video_id(df_mapping, video_id)
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
                result = values_class.find_values_with_video_id(df_mapping, video_id)
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

    def is_rider_id(self, df, id, avg_height, min_shared_frames=5,
                    dist_thresh=80, similarity_thresh=0.8, overlap_ratio=0.7):
        """
        Determines if a person identified by the given unique-id is riding a bicycle or motorcycle
        during their trajectory in the YOLO detection DataFrame.

        The function checks, for the duration in which the person is present, whether a bicycle or
        motorcycle detection is present and moving together (i.e., close proximity and similar
        movement direction and speed) with the person for a sufficient number of frames. If so,
        the person is likely a cyclist or motorcyclist and should be excluded from pedestrian analysis.

        Args:
            df (pd.DataFrame): YOLO detections DataFrame containing columns:
                'yolo-id' (class, 0=person, 1=bicycle, 3=motorcycle), 'unique-id',
                'frame-count', 'x-center', 'y-center', 'width', 'height'.
            id (int or str): The unique-id of the person to analyse.
            avg_height (float): The average real-world height of the person (cm).
            min_shared_frames (int, optional): Minimum number of frames with both the person and vehicle
                present for comparison. Defaults to 5.
            dist_thresh (float, optional): Maximum distance (in pixels) between person and vehicle
                bounding box centers to be considered "moving together". Defaults to 50.
            similarity_thresh (float, optional): Minimum cosine similarity threshold for movement direction
                to be considered "similar". Ranges from -1 (opposite) to 1 (identical direction). Defaults to 0.8.
            overlap_ratio (float, optional): Fraction of overlapping frames where proximity and movement
                similarity must be satisfied. Defaults to 0.7 (i.e., 70%).

        Returns:
            bool: True if the person is moving together with a bicycle or motorcycle (i.e., is a rider);
                False if likely a pedestrian.

        Example:
            for id, time in id_time.items():
                if is_rider_id(df, id):
                    continue  # Skip this id (cyclist or motorcyclist)
                # ...process as pedestrian...

        """
        # Extract all rows corresponding to the person id
        person_track = df[df['unique-id'] == id]
        if person_track.empty:
            return False  # No data for this id

        frames = person_track['frame-count'].values
        if len(frames) < min_shared_frames:
            return False  # Not enough frames to perform check

        first_frame, last_frame = frames.min(), frames.max()

        # Filter DataFrame to get all bicycle/motorcycle detections in relevant frames
        mask = (
            (df['frame-count'] >= first_frame)
            & (df['frame-count'] <= last_frame)
            & (df['yolo-id'].isin([1, 3]))
        )
        vehicles_in_frames = df[mask]

        for vehicle_id in vehicles_in_frames['unique-id'].unique():
            # Get trajectory for this vehicle
            vehicle_track = vehicles_in_frames[vehicles_in_frames['unique-id'] == vehicle_id]

            # Find shared frames between person and vehicle
            shared_frames = np.intersect1d(person_track['frame-count'], vehicle_track['frame-count'])

            if len(shared_frames) < min_shared_frames:
                continue  # Not enough overlapping frames to check movement together

            # Align positions for person and vehicle on shared frames, sorted by Frame Count
            person_pos = (
                person_track[person_track['frame-count'].isin(shared_frames)]
                .sort_values('frame-count')[['x-center', 'y-center']].values
            )
            vehicle_pos = (
                vehicle_track[vehicle_track['frame-count'].isin(shared_frames)]
                .sort_values('frame-count')[['x-center', 'y-center']].values
            )

            # Calculate person's bounding box heights in pixels for shared frames
            person_heights = (
                person_track[person_track['frame-count'].isin(shared_frames)]
                .sort_values('frame-count')['height'].values
            )  # This is in pixels per frame

            # Compute pixels-per-cm for each frame using the average real-world height
            pixels_per_cm = person_heights / avg_height  # array, one per frame

            # Compute Euclidean distances between person and vehicle centers for shared frames
            pixel_dists = np.linalg.norm(person_pos - vehicle_pos, axis=1)
            distances_cm = pixel_dists / pixels_per_cm  # Real-world distance per frame

            proximity = (distances_cm < dist_thresh)
            if proximity.sum() / len(distances_cm) < overlap_ratio:
                continue  # Not close enough for sufficient frames

            # Calculate movement vectors (delta positions) for both tracks
            person_mov = np.diff(person_pos, axis=0)
            vehicle_mov = np.diff(vehicle_pos, axis=0)

            # Compute cosine similarity of movement direction for each step
            similarities = []
            for a, b in zip(person_mov, vehicle_mov):
                if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                    similarities.append(0)
                else:
                    similarities.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            similarity_mask = (np.array(similarities) > similarity_thresh)

            if similarity_mask.sum() / len(similarities) >= overlap_ratio:
                return True

        # If no such vehicle found moving together, label as pedestrian
        return False

    def is_valid_crossing(self, df, person_id, ratio_thresh=0.2):
        """
        Determines if a detected pedestrian crossing is valid based on the relative movement
        between the person and static objects (traffic light or stop sign) in the YOLO output.

        The function assumes the camera (e.g., dashcam) may be moving. To account for this,
        it compares the y-direction movement of the static object(s) and the person. If the
        static object's y-movement is a significant fraction of the person's y-movement, it
        is likely due to camera movement rather than actual pedestrian crossing, and the
        crossing is classified as invalid.

        Args:
            df (pd.DataFrame): DataFrame containing YOLO detections with columns:
                'yolo-id', 'x-center', 'y-center', 'width', 'height', 'unique-id', 'frame-count'.
                All coordinates are normalized between 0 and 1.
            person_id (int or str): Unique Id of the person (pedestrian) to be analyzed.
            key: Placeholder for additional parameters (not used in this function).
            avg_height: Placeholder for additional parameters (not used in this function).
            fps: Placeholder for additional parameters (not used in this function).
            ratio_thresh (float, optional): Threshold ratio of static object y-movement to
                person's y-movement. Default is 0.2.

        Returns:
            bool: True if crossing is valid (not due to camera movement), False otherwise.
        """

        # Extract all detections for the specified person
        person_track = df[df['unique-id'] == person_id]
        if person_track.empty:
            # No detection for this person
            return False

        # Determine the first and last frames in which the person appears
        frames = person_track['frame-count'].values
        first_frame, last_frame = frames.min(), frames.max()

        # Filter for static objects (traffic light or stop sign) in those frames
        static_objs = df[
            (df['frame-count'] >= first_frame) &
            (df['frame-count'] <= last_frame) &
            (df['yolo-id'].isin([9, 11]))
        ]

        if static_objs.empty:
            # No static object present during this period; cannot verify, assume valid
            return True

        # Calculate the y-movement (vertical movement) of the person
        person_y_movement = person_track['y-center'].max() - person_track['y-center'].min()

        # Calculate the maximum y-movement among all static objects in the same frame range
        max_static_y_movement = 0
        for obj_id in static_objs['unique-id'].unique():
            obj_track = static_objs[static_objs['unique-id'] == obj_id]
            y_movement = obj_track['y-center'].max() - obj_track['y-center'].min()
            if y_movement > max_static_y_movement:
                max_static_y_movement = y_movement

        # If the person does not move vertically, it's an invalid crossing
        if person_y_movement == 0:
            return False

        # Compute the movement ratio
        movement_ratio = max_static_y_movement / person_y_movement

        # If the static object moves too much compared to the person, crossing is invalid
        if movement_ratio > ratio_thresh:
            return False
        else:
            return True
