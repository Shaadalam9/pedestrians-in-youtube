import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
from .values import Values
from .wrappers import Wrappers
import pandas as pd
from tqdm import tqdm

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

values_class = Values()
wrapper_class = Wrappers()

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
            (video, start, end, time_of_day, city, state, country, gdp_, population, population_country,
             traffic_mortality_, continent, literacy_rate, avg_height, iso3, fps) = result

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

    def calculate_speed_of_crossing(self, df_mapping, dfs, data):
        """
        Calculate the walking speeds of individuals crossing in various videos
        and organise them by city, state, and condition.

        Args:
            df_mapping (dataframe): A DataFrame mapping video IDs to metadata such as
                city, state, country, and other contextual information.
            dfs (dict): A dictionary where contains all the csv files extracted from YOLO.
            data (dict): A dictionary where keys are video IDs and values are dictionaries
                mapping person IDs to crossing durations.

        Returns:
            speed_dict (dict): A dictionary with keys formatted as 'city_state_condition' mapping to lists
                of walking speeds (m/s) for each valid crossing.
            all_speed (list): A flat list of all calculated walking speeds (m/s) across videos, including outliers.
        """
        time_ = []
        speed_compelete = {}

        # Create a dictionary to store country information for each city
        city_country_map_ = {}

        # Iterate over each video data
        for key, df in tqdm(data.items(), total=len(data)):
            speed_id_compelete = {}
            if df == {}:  # Skip if there is no data
                continue
            result = values_class.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                (_, start, end, condition, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso3, fps) = result

                value = dfs.get(key)

                # Store the country associated with each city
                city_country_map_[f'{city}_{state}'] = iso3

                # Calculate the duration of the video
                duration = end - start
                time_.append(duration)

                grouped = value.groupby('Unique Id')
                for id, time in df.items():
                    grouped_with_id = grouped.get_group(id)
                    mean_height = grouped_with_id['Height'].mean()
                    min_x_center = grouped_with_id['X-center'].min()
                    max_x_center = grouped_with_id['X-center'].max()

                    ppm = mean_height / avg_height
                    distance = (max_x_center - min_x_center) / ppm

                    speed_ = (distance / time) / 100
                    speed_id_compelete[id] = speed_

                    # Taken from https://www.wikiwand.com/en/articles/Preferred_walking_speed
                    # if speed_ > 1.42:  # Exclude outlier speeds

                speed_compelete[key] = speed_id_compelete

        output = wrapper_class.city_wrapper(input_dict=speed_compelete, mapping=df_mapping)

        return output

    def avg_speed_of_crossing(self, df_mapping, dfs, data):
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

        speed_dict = self.calculate_speed_of_crossing(df_mapping, dfs, data)

        for city_condition, value_1 in speed_dict.items():
            box = []
            for video_id, value_2 in value_1.items():
                for unique_id, speed in value_2.items():
                    box.append(speed)
            avg_speed[city_condition] = (sum(box) / len(box))

        return avg_speed

    def time_to_start_cross(self, df_mapping, dfs, data, person_id=0):
        """
        Calculate the time to start crossing the road of individuals crossing in various videos
        and organise them by city, state, and condition.

        Args:
            df_mapping (dataframe): A DataFrame mapping video IDs to metadata such as
                city, state, country, and other contextual information.
            dfs (dict): A dictionary where contains all the csv files extracted from YOLO.
            data (dict): A dictionary where keys are video IDs and values are dictionaries
                mapping person IDs to crossing durations.
            person_id (int, optional): YOLO unique representation for person

        Returns:
            speed_dict (dict): A dictionary with keys formatted as 'city_state_condition' mapping to lists
                of walking speeds (m/s) for each valid crossing.
            all_speed (list): A flat list of all calculated walking speeds (m/s) across videos, including outliers.
        """
        time_compelete = {}

        for key, df in tqdm(dfs.items(), total=len(dfs)):
            data_cross = {}
            time_id_complete = {}
            crossed_ids = df[(df["YOLO_id"] == person_id)]

            # Extract relevant information using the find_values function
            result = values_class.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                (_, start, end, condition, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso3, fps) = result

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
                    continue

                time_compelete[key] = time_id_complete

        output = wrapper_class.city_wrapper(input_dict=time_compelete, mapping=df_mapping)

        return output

    def avg_time_to_start_cross(self, df_mapping, dfs, data):
        avg_over_time = {}

        output = self.time_to_start_cross(df_mapping, dfs, data)

        for city_condition, value_1 in output.items():
            box = []
            for video_id, value_2 in value_1.items():
                if value_2 is None:
                    continue
                else:
                    result = values_class.find_values_with_video_id(df_mapping, video_id)
                    if result is not None:
                        (_, start, end, condition, city, state, country, gdp_, population, population_country,
                         traffic_mortality_, continent, literacy_rate, avg_height, iso3, fps) = result
                        for unique_id, time in value_2.items():
                            if time > 0:
                                box.append(time/(fps/10))

            avg_over_time[city_condition] = (sum(box) / len(box))

        return avg_over_time
