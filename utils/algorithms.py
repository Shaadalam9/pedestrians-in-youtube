# by Shadab Alam <md_shadab_alam@outlook.com>
import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
from utils.values import Values
from utils.wrappers import Wrappers
import pandas as pd
import numpy as np
import os
from helper_script import Youtube_Helper
# from analysis import Analysis

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

values_class = Values()
wrapper_class = Wrappers()
helper = Youtube_Helper()
# analysis_class = Analysis()

df_mapping = pd.read_csv(common.get_configs("mapping"))


class Algorithms():
    def __init__(self) -> None:
        pass

    def time_to_cross(self, dataframe, ids, video_id):
        """Calculates the time taken for each object with specified IDs to cross the road.

        Args:
            dataframe (DataFrame): The DataFrame (csv file) containing object data.
            ids (list): A list of unique IDs of objects which are crossing the road.

        Returns:
            dict: A dictionary where keys are object IDs and values are the time taken for
            each object to cross the road, in seconds.
        """
        if 'Frame Count' not in dataframe.columns:
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
            x_min = dataframe[dataframe["Unique Id"] == id]["X-center"].min()
            x_max = dataframe[dataframe["Unique Id"] == id]["X-center"].max()

            # Get a sorted group of entries for the current object ID
            sorted_grp = dataframe[dataframe["Unique Id"] == id]

            # Get the corresponding frame counts instead of index
            x_min_frame = sorted_grp[sorted_grp['X-center'] == x_min]['Frame Count'].iloc[0]
            x_max_frame = sorted_grp[sorted_grp['X-center'] == x_max]['Frame Count'].iloc[0]

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
        time_ = []  # List to store durations of videos (not used in output)
        speed_compelete = {}  # Dictionary to hold valid speed results for each video

        # Create a dictionary to store country information for each city
        city_country_map_ = {}

        # Group YOLO data by unique person ID
        grouped = df.groupby('Unique Id')

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
                fps = result[17]

                # value = dfs.get(key)  # Get corresponding YOLO data

                # Store the country associated with each city
                city_country_map_[f'{city}_{lat}_{long}'] = iso3

                # Calculate total duration of the crossing segment in this video
                duration = end - start
                time_.append(duration)

                # Loop through each individual's crossing data in this video
                for id, time in id_time.items():

                    if self.is_rider_id(df, id, key, avg_height, fps):
                        continue  # Skip rider, not a pedestrian

                    # Get all frames for this person
                    grouped_with_id = grouped.get_group(id)

                    # Calculate mean height of bounding box for this person
                    mean_height = grouped_with_id['Height'].mean()

                    # Find minimum and maximum X-center positions to estimate path length
                    min_x_center = grouped_with_id['X-center'].min()
                    max_x_center = grouped_with_id['X-center'].max()

                    # Estimate "pixels per centi-meter" using average height and actual avg_height
                    ppm = mean_height / avg_height

                    # Estimate real-world distance crossed (in centimeters)
                    distance = (max_x_center - min_x_center) / ppm

                    # Calculate walking speed (meters per second)
                    speed_ = (distance / time) / 100

                    # Only include the speed if it's within configured bounds
                    if common.get_configs("min_speed_limit") <= speed_ <= common.get_configs("max_speed_limit"):
                        speed_id_compelete[id] = speed_
                    # Otherwise, skip this crossing as an outlier

                # Store all valid speeds for this video
                speed_compelete[key] = speed_id_compelete

        # Group and organise the results for downstream analysis/plotting
        output = wrapper_class.city_country_wrapper(input_dict=speed_compelete, mapping=df_mapping)

        return output

    def avg_speed_of_crossing(self, all_speed):
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
        avg_speed = {}

        for city_condition, value_1 in all_speed.items():
            box = []
            for video_id, value_2 in value_1.items():
                for unique_id, speed in value_2.items():
                    box.append(speed)
            if len(box) > 0:
                avg_speed[city_condition] = (sum(box) / len(box))

        return avg_speed

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

        time_compelete = {}

        if 'Frame Count' not in df.columns:
            return

        data_cross = {}
        time_id_complete = {}

        # Filter out the person ids for faster processing
        crossed_ids = df[(df["YOLO_id"] == person_id)]

        # Extract relevant information using the find_values function
        result = values_class.find_values_with_video_id(df_mapping, next(iter(data)))

        # Check if the result is None (i.e., no matching data was found)
        if result is not None:

            # Makes group based on Unique ID
            crossed_ids_grouped = crossed_ids.groupby("Unique Id")

            for unique_id, group_data in crossed_ids_grouped:
                x_values = group_data["X-center"].values
                initial_x = x_values[0]  # Initial x-value
                mean_height = group_data['Height'].mean()
                flag = 0
                margin = 0.1 * mean_height  # Margin for considering crossing event
                consecutive_frame = 0

                for i in range(0, len(x_values)-10, 10):
                    if initial_x < 0.5:  # Check if crossing from left to right
                        if (x_values[i] - margin <= x_values[i+10] <= x_values[i] + margin):
                            consecutive_frame += 1
                            if consecutive_frame == 3:  # Check for three consecutive frames
                                flag = 1
                        elif flag == 1:
                            data_cross[unique_id] = consecutive_frame
                            break
                        else:
                            consecutive_frame = 0

                    else:  # Check if crossing from right to left
                        if (x_values[i] - margin >= x_values[i+10] >= x_values[i] + margin):
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
                return

            time_compelete[next(iter(data))] = time_id_complete

        output = wrapper_class.city_country_wrapper(input_dict=time_compelete, mapping=df_mapping)

        return output

    def avg_time_to_start_cross(self, df_mapping, all_time):
        avg_over_time = {}

        for city_condition, value_1 in all_time.items():
            box = []
            for video_id, value_2 in value_1.items():
                if value_2 is None:
                    continue
                else:
                    result = values_class.find_values_with_video_id(df_mapping, video_id)
                    if result is not None:
                        fps = result[17]
                        for unique_id, time in value_2.items():
                            if time > 0:
                                box.append(time/(fps/10))

            avg_over_time[city_condition] = (sum(box) / len(box))

        return avg_over_time

    def is_rider_id(self, df, id, key, avg_height, fps, min_shared_frames=5,
                    dist_thresh=80, similarity_thresh=0.8, overlap_ratio=0.7):
        """
        Determines if a person identified by the given Unique Id is riding a bicycle or motorcycle
        during their trajectory in the YOLO detection DataFrame.

        The function checks, for the duration in which the person is present, whether a bicycle or
        motorcycle detection is present and moving together (i.e., close proximity and similar
        movement direction and speed) with the person for a sufficient number of frames. If so,
        the person is likely a cyclist or motorcyclist and should be excluded from pedestrian analysis.

        Args:
            df (pd.DataFrame): YOLO detections DataFrame containing columns:
                'YOLO_id' (class, 0=person, 1=bicycle, 3=motorcycle), 'Unique Id',
                'Frame Count', 'X-center', 'Y-center', 'Width', 'Height'.
            id (int or str): The Unique Id of the person to analyze.
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
        person_track = df[df['Unique Id'] == id]
        if person_track.empty:
            return False  # No data for this id

        frames = person_track['Frame Count'].values
        if len(frames) < min_shared_frames:
            return False  # Not enough frames to perform check

        first_frame, last_frame = frames.min(), frames.max()

        # Filter DataFrame to get all bicycle/motorcycle detections in relevant frames
        mask = (
            (df['Frame Count'] >= first_frame)
            & (df['Frame Count'] <= last_frame)
            & (df['YOLO_id'].isin([1, 3]))
        )
        vehicles_in_frames = df[mask]

        for vehicle_id in vehicles_in_frames['Unique Id'].unique():
            # Get trajectory for this vehicle
            vehicle_track = vehicles_in_frames[vehicles_in_frames['Unique Id'] == vehicle_id]

            # Find shared frames between person and vehicle
            shared_frames = np.intersect1d(person_track['Frame Count'], vehicle_track['Frame Count'])

            if len(shared_frames) < min_shared_frames:
                continue  # Not enough overlapping frames to check movement together

            # Align positions for person and vehicle on shared frames, sorted by Frame Count
            person_pos = (
                person_track[person_track['Frame Count'].isin(shared_frames)]
                .sort_values('Frame Count')[['X-center', 'Y-center']].values
            )
            vehicle_pos = (
                vehicle_track[vehicle_track['Frame Count'].isin(shared_frames)]
                .sort_values('Frame Count')[['X-center', 'Y-center']].values
            )

            # Calculate person's bounding box heights in pixels for shared frames
            person_heights = (
                person_track[person_track['Frame Count'].isin(shared_frames)]
                .sort_values('Frame Count')['Height'].values
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
                # If both proximity and movement similarity criteria met, label as rider

                video_name, start_offset = key.rsplit('_', 1)

                # Find the existing folder containing the video file
                existing_folder = next((
                    path for path in common.get_configs("videos") if os.path.exists(
                        os.path.join(path, f"{video_name}.mp4"))), None)

                if not existing_folder:
                    raise FileNotFoundError(f"Video file '{video_name}.mp4' not found in any of the specified paths.")  # noqa:E501

                base_video_path = os.path.join(existing_folder, f"{video_name}.mp4")

                # Look up the frame rate (fps) using the video_start_time
                result = values_class.find_values_with_video_id(df_mapping, key)

                # Check if the result is None (i.e., no matching data was found)
                if result is not None:
                    # Unpack the result since it's not None

                    first_time = first_frame / fps
                    last_time = last_frame / fps

                    # Adjusted start and end times
                    real_start_time = first_time + float(start_offset)
                    real_end_time = float(start_offset) + last_time

                    # Filter dataframe for only the shared frames of this person and vehicle
                    filtered_df = df[
                        ((df['Unique Id'] == id) | (df['Unique Id'] == vehicle_id))
                        & (df['Frame Count'].isin(shared_frames))
                    ]

                    # Trim and save the raw segment
                    helper.trim_video(
                        input_path=base_video_path,
                        output_path=os.path.join("saved_snaps", "original", f"{video_name}_{real_start_time}.mp4"),
                        start_time=real_start_time, end_time=real_end_time)
                    helper.draw_yolo_boxes_on_video(df=filtered_df, fps=fps,
                                                    video_path=os.path.join("saved_snaps",
                                                                            "original",
                                                                            f"{video_name}_{real_start_time}.mp4"),
                                                    output_path=os.path.join("saved_snaps",
                                                                             "tracked",
                                                                             f"{video_name}_{real_start_time}.mp4"))

                return True

        # If no such vehicle found moving together, label as pedestrian
        return False
