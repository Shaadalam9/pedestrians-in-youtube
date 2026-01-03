import os
import common
from tqdm import tqdm
import pandas as pd
from custom_logger import CustomLogger
from utils.core.metadata import MetaData

metadata_class = MetaData()
logger = CustomLogger(__name__)  # use custom logger


class Events:
    def __init__(self) -> None:
        pass

    @staticmethod
    def crossing_event_with_traffic_equipment(df_mapping, data):
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
            count_with_equipment = 0
            count_without_equipment = 0

            # Extract metadata for this video
            result = metadata_class.find_values_with_video_id(df_mapping, video_key)
            if result is not None:
                start_time = result[1]
                end_time = result[2]
                condition = result[3]
                city = result[4]
                latitude = result[6]
                longitude = result[7]
                country = result[8]

                location_key_city = f'{city}_{latitude}_{longitude}_{condition}'
                location_key_country = f'{country}_{condition}'

                # Update total duration per location/condition
                duration = end_time - start_time
                total_duration_by_city[location_key_city] = total_duration_by_city.get(location_key_city, 0) + duration
                total_duration_by_country[location_key_country] = total_duration_by_country.get(
                    location_key_country, 0) + duration

                # Find the CSV file for this video
                value = None
                for folder_path in common.get_configs('data'):
                    for subfolder in common.get_configs("sub_domain"):
                        subfolder_path = os.path.join(folder_path, subfolder)
                        if not os.path.exists(subfolder_path):
                            continue  # Skip if subfolder doesn't exist

                        for file in os.listdir(subfolder_path):
                            if os.path.splitext(file)[0] == video_key:
                                file_path = os.path.join(subfolder_path, file)
                                value = pd.read_csv(file_path)

                                # Keep only rows with confidence > min_conf
                                value = value[value["confidence"] >= common.get_configs("min_confidence")]

                                break
                        if value is not None:
                            break  # Break out of subfolder loop
                    if value is not None:
                        break  # Break out of folder_path loop

                if value is None:
                    continue  # Skip if file not found

                # Analyse crossings for presence of traffic equipment
                for unique_id, _ in crossings.items():
                    unique_id_indices = value.index[value['unique-id'] == unique_id]
                    if unique_id_indices.empty:
                        continue  # Skip if no occurrences

                    first_occurrence = unique_id_indices[0]
                    last_occurrence = unique_id_indices[-1]

                    yolo_ids = value.loc[first_occurrence:last_occurrence, 'yolo-id']

                    has_equipment = yolo_ids.isin([9, 11]).any()
                    lacks_equipment = not yolo_ids.isin([9, 11]).any()

                    if has_equipment:
                        count_with_equipment += 1
                    if lacks_equipment:
                        count_without_equipment += 1

                # Aggregate by city/condition
                crossings_with_traffic_equipment_city[location_key_city] = \
                    crossings_with_traffic_equipment_city.get(location_key_city, 0) + count_with_equipment
                crossings_without_traffic_equipment_city[location_key_city] = \
                    crossings_without_traffic_equipment_city.get(location_key_city, 0) + count_without_equipment

                # Aggregate by country/condition
                crossings_with_traffic_equipment_country[location_key_country] = \
                    crossings_with_traffic_equipment_country.get(location_key_country, 0) + count_with_equipment
                crossings_without_traffic_equipment_country[location_key_country] = \
                    crossings_without_traffic_equipment_country.get(location_key_country, 0) + count_without_equipment

        return (crossings_with_traffic_equipment_city,
                crossings_without_traffic_equipment_city,
                total_duration_by_city,
                crossings_with_traffic_equipment_country,
                crossings_without_traffic_equipment_country,
                total_duration_by_country)

    # TODO: combine methods for looking at crossing events with/without traffic lights
    @staticmethod
    def crossing_event_wt_traffic_light(df_mapping, data):
        """Plots traffic mortality rate vs percentage of crossing events without traffic light.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
            data (dict): Dictionary containing pedestrian crossing data.
        """
        var_exist, var_nt_exist, ratio = {}, {}, {}
        time_ = []

        counter_1, counter_2 = {}, {}

        # For a specific id of a person search for the first and last occurrence of that id and see if the traffic
        # light was present between it or not. Only getting those unique_id of the person who crosses the road.

        # Loop through each video data
        for key, df in tqdm(data.items(), total=len(data)):
            counter_exists, counter_nt_exists = 0, 0

            # Extract relevant information using the find_values function
            result = metadata_class.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                start = result[1]
                end = result[2]
                condition = result[3]
                city = result[4]
                lat = result[6]
                long = result[7]

                # Calculate the duration of the video
                duration = end - start
                time_.append(duration)

                for folder_path in common.get_configs('data'):
                    existing_subfolders = []
                    for subfolder in common.get_configs("sub_domain"):
                        subfolder_path = os.path.join(folder_path, subfolder)
                        if os.path.exists(subfolder_path):
                            existing_subfolders.append(subfolder)

                    # If none of the subfolders exist, print/log once
                    if not existing_subfolders:
                        logger.warning(f"None of the subfolders ('bbox', 'seg') exist in {folder_path}.")
                        continue

                    for subfolder in existing_subfolders:
                        subfolder_path = os.path.join(folder_path, subfolder)
                        for file in os.listdir(subfolder_path):
                            filename_no_ext = os.path.splitext(file)[0]
                            if filename_no_ext == key:
                                file_path = os.path.join(subfolder_path, file)
                                # Load the CSV
                                value = pd.read_csv(file_path)

                                # Keep only rows with confidence > min_conf
                                value = value[value["confidence"] >= common.get_configs("min_confidence")]

                for id, time in df.items():
                    unique_id_indices = value.index[value['unique-id'] == id]
                    first_occurrence = unique_id_indices[0]
                    last_occurrence = unique_id_indices[-1]

                    # Check if yolo-id = 9 exists within the specified index range
                    yolo_id_9_exists = any(
                        value.loc[first_occurrence:last_occurrence, 'yolo-id'] == 9)
                    yolo_id_9_not_exists = not any(
                        value.loc[first_occurrence:last_occurrence, 'yolo-id'] == 9)

                    if yolo_id_9_exists:
                        counter_exists += 1
                    if yolo_id_9_not_exists:
                        counter_nt_exists += 1

                # Normalising the counters
                var_exist[key] = ((counter_exists * 60) / time_[-1])
                var_nt_exist[key] = ((counter_nt_exists * 60) / time_[-1])
                city_id_format = f'{city}_{lat}_{long}_{condition}'

                counter_1[city_id_format] = counter_1.get(city_id_format, 0) + var_exist[key]
                counter_2[city_id_format] = counter_2.get(city_id_format, 0) + var_nt_exist[key]

                if (counter_1[city_id_format] + counter_2[city_id_format]) == 0:
                    # Gives an error of division by 0
                    continue
                else:
                    if city_id_format in ratio:
                        ratio[city_id_format] = ((counter_2[city_id_format] * 100) /
                                                 (counter_1[city_id_format] +
                                                  counter_2[city_id_format]))
                        continue
                    # If already present, the array below will be filled multiple times
                    else:
                        ratio[city_id_format] = ((counter_2[city_id_format] * 100) /
                                                 (counter_1[city_id_format] +
                                                  counter_2[city_id_format]))
        return ratio
