# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import math
import pandas as pd
import numpy as np
import os
import cv2
from collections import defaultdict
import heapq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helper_script import Youtube_Helper
from utils.algorithms import Algorithms
from utils.values import Values
from utils.wrappers import Wrappers
from utils.plot import Plots
from utils.geometry import Geometry
import common
from custom_logger import CustomLogger
from logmod import logs
import statistics
import ast
import pickle
from tqdm import tqdm
import re
import warnings
from scipy.spatial import KDTree
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from datetime import datetime

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

helper = Youtube_Helper()
algorithms_class = Algorithms()
values_class = Values()
wrapper_class = Wrappers()
plots_class = Plots()
geometry_class = Geometry()

# set template for plotly output
template = common.get_configs('plotly_template')

# File to store the city coordinates
file_results = 'results.pickle'

# Colours in graphs
bar_colour_1 = 'rgb(251, 180, 174)'
bar_colour_2 = 'rgb(179, 205, 227)'
bar_colour_3 = 'rgb(204, 235, 197)'
bar_colour_4 = 'rgb(222, 203, 228)'

# Consts
BASE_HEIGHT_PER_ROW = 20  # Adjust as needed
FLAG_SIZE = 12
TEXT_SIZE = 12
SCALE = 1  # scale=3 hangs often

video_paths = common.get_configs("videos")


class Analysis():

    def __init__(self) -> None:
        pass

    @staticmethod
    def read_csv_files(folder_paths, df_mapping):
        """
        Reads all CSV files in the specified folders, processes them if configured,
        and returns their contents as a dictionary keyed by file name.

        This function will:
          - Iterate over the provided list of folder paths.
          - For each folder, it will check if it exists and log a warning if not.
          - For each CSV file found, it will read the file into a pandas DataFrame.
          - Optionally apply geometry correction to the DataFrame if enabled in the configuration.
          - Look up additional values using the provided mapping and the file's base name.
          - Adds the DataFrame to the results only if:
                - The mapping values exist and the population of the city is greater than the
                    configured `footage_threshold,
                - The total seconds are greater than the configured `footage_threshold`.

        Args:
            folder_paths (list[str]): List of folder paths containing the CSV files.
            df_mapping (Any): A mapping object used to find values related to each file (for example, video IDs).

        Returns:
            dict: Dictionary where keys are the base file names (without extension),
                  and values are the corresponding pandas DataFrames of each CSV file.
                  Only files meeting all value requirements are included.
        """
        dfs = {}
        logger.info("Reading csv files.")

        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                logger.warning(f"Folder does not exist: {folder_path}.")
                continue

            for file in tqdm(os.listdir(folder_path)):
                if file.endswith(".csv"):
                    filename = os.path.splitext(file)[0]

                    # Lookup values *before* reading CSV
                    values = values_class.find_values_with_video_id(df_mapping, filename)
                    if values is None:
                        continue  # Skip if mapping or required value is None

                    total_seconds = values_class.calculate_total_seconds_for_city(
                        df_mapping, values[4], values[5]
                    )
                    if total_seconds <= common.get_configs("footage_threshold"):
                        continue  # Skip if not enough seconds

                    file_path = os.path.join(folder_path, file)
                    try:
                        logger.debug(f"Adding file {file_path} to dfs.")

                        # Read the CSV into a DataFrame
                        df = pd.read_csv(file_path)

                        # Optionally apply geometry correction if configured and not zero
                        use_geom_correction = common.get_configs("use_geometry_correction")
                        if use_geom_correction != 0:
                            df = geometry_class.reassign_ids_directional_cross_fix(
                                df,
                                distance_threshold=use_geom_correction,
                                yolo_ids=[0]
                            )

                        # Add the DataFrame to the dict
                        dfs[filename] = df
                    except Exception as e:
                        logger.error(f"Failed to read {file_path}: {e}.")
                        continue  # Skip to the next file if reading fails
        return dfs

    @staticmethod
    def count_object(dataframe, id):
        """
        Counts the number of unique instances of an object with a specific ID in a DataFrame.

        Args:
            dataframe (DataFrame): The DataFrame containing object data.
            id (int): The unique ID assigned to the object.

        Returns:
            int: The number of unique instances of the object with the specified ID.
        """

        # Filter the DataFrame to include only entries for the specified object ID
        crossed_ids = dataframe[(dataframe["YOLO_id"] == id)]

        # Group the filtered data by Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

        # Count the number of groups, which represents the number of unique instances of the object
        num_groups = crossed_ids_grouped.ngroups

        return num_groups

    @staticmethod
    def calculate_total_seconds(df):
        """Calculates the total seconds of the total video according to mapping file."""
        grand_total_seconds = 0

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Extracting data from the DataFrame row

            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])

            # Iterate through each start time and end time
            for start, end in zip(start_times, end_times):
                for s, e in zip(start, end):
                    grand_total_seconds += (int(e) - int(s))

        return grand_total_seconds

    @staticmethod
    def calculate_total_videos(df):
        """
        Calculates the total number of videos in the mapping file.
        """
        total_videos = set()
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            videos = row["videos"]

            videos_list = videos.split(",")  # Split by comma to convert string to list

            for video in videos_list:
                total_videos.add(video.strip())  # Add the video to the set (removing any extra whitespace)

        return len(total_videos)

    @staticmethod
    def get_unique_values(df, value):
        """
        Calculates the number of unique countries from a DataFrame.

        Args:
            df (DataFrame): A DataFrame containing the CSV data.

        Returns:
            tuple: A set of unique countries and the total count of unique countries.
        """
        unique_values = set(df[value].unique())

        return unique_values, len(unique_values)

    @staticmethod
    def get_world_map(df_mapping):
        """
        Generate a world map with highlighted countries and red markers for cities using Plotly.

        - Highlights countries based on the cities present in the dataset.
        - Adds scatter points for each city with detailed socioeconomic and traffic-related hover info.
        - Adjusts map appearance to improve clarity and remove irrelevant regions like Antarctica.

        Args:
            df_mapping (pd.DataFrame): A DataFrame with columns:
                ['city', 'state', 'country', 'lat', 'lon', 'continent',
                 'gmp', 'population_city', 'population_country',
                 'traffic_mortality', 'literacy_rate', 'avg_height',
                 'gini', 'traffic_index']

        Returns:
            None. Saves and displays the interactive map to disk.
        """
        cities = df_mapping["city"]
        states = df_mapping["state"]
        countries = df_mapping["country"]
        coords_lat = df_mapping["lat"]
        coords_lon = df_mapping["lon"]

        # Create the country list to highlight in the choropleth map
        countries_set = set(countries)  # Use set to avoid duplicates
        if "Denmark" in countries_set:
            countries_set.add('Greenland')
        if "Turkiye" in countries_set:
            countries_set.add('Turkey')

        # Create a DataFrame for highlighted countries with a value (same for all to have the same color)
        df = pd.DataFrame({'country': list(countries_set), 'value': 1})

        # Create a choropleth map using Plotly with grey color for countries
        fig = px.choropleth(df, locations="country", locationmode="country names",
                            color="value", hover_name="country", hover_data={'value': False, 'country': False},
                            color_continuous_scale=["rgb(242, 186, 78)", "rgb(242, 186, 78)"],
                            labels={'value': 'Highlighted'})

        # Update layout to remove Antarctica, Easter Island, remove the color bar, and set ocean color
        fig.update_layout(
            coloraxis_showscale=False,  # Remove color bar
            geo=dict(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="black",  # Set coastline color
                showcountries=True,  # Show country borders
                countrycolor="black",  # Set border color
                projection_type='equirectangular',
                showlakes=True,
                lakecolor='rgb(173, 216, 230)',  # Light blue for lakes
                projection_scale=1,
                center=dict(lat=20, lon=0),  # Center map to remove Antarctica
                bgcolor='rgb(173, 216, 230)',  # Light blue for ocean
                resolution=50
            ),
            margin=dict(l=0, r=0, t=0, b=0),  # Remove the margins
            paper_bgcolor='rgb(173, 216, 230)'  # Set the paper background to match the ocean color
        )

        # Process each city and its corresponding country
        city_coords = []
        for i, (city, state, lat, lon) in enumerate(tqdm(zip(cities, states, coords_lat, coords_lon), total=len(cities))):  # noqa: E501
            if not state or str(state).lower() == 'nan':
                state = 'N/A'
            if lat and lon:
                city_coords.append({
                    'City': city,
                    'State': state,
                    'Country': df_mapping["country"].iloc[i],
                    'Continent': df_mapping["continent"].iloc[i],
                    'lat': lat,
                    'lon': lon,
                    'GDP (Billion USD)': df_mapping["gmp"].iloc[i],
                    'City population (thousands)': df_mapping["population_city"].iloc[i] / 1000.0,
                    'Country population (thousands)': df_mapping["population_country"].iloc[i] / 1000.0,
                    'Traffic mortality rate (per 100,000)': df_mapping["traffic_mortality"].iloc[i],
                    'Literacy rate': df_mapping["literacy_rate"].iloc[i],
                    'Average height (cm)': df_mapping["avg_height"].iloc[i],
                    'Gini coefficient': df_mapping["gini"].iloc[i],
                    'Traffic index': df_mapping["traffic_index"].iloc[i],
                })

        if city_coords:
            city_df = pd.DataFrame(city_coords)
            # city_df["City"] = city_df["city"]  # Format city name with "City:"
            city_trace = px.scatter_geo(
                city_df, lat='lat', lon='lon',
                hover_data={
                    'City': True,
                    'State': True,
                    'Country': True,
                    'Continent': True,
                    'GDP (Billion USD)': True,
                    'City population (thousands)': True,
                    'Country population (thousands)': True,
                    'Traffic mortality rate (per 100,000)': True,
                    'Literacy rate': True,
                    'Average height (cm)': True,
                    'Gini coefficient': True,
                    'Traffic index': True,
                    'lat': False,
                    'lon': False  # Hide lat and lon
                }
            )
            # Update the city markers to be red and adjust size
            city_trace.update_traces(marker=dict(color="red", size=5))

            # Add the scatter_geo trace to the choropleth map
            fig.add_trace(city_trace.data[0])

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Save and display the figure
        plots_class.save_plotly_figure(fig, "world_map", save_final=True)

    @staticmethod
    def get_mapbox_map(df, hover_data=None, file_name="mapbox_map"):
        """Generate world map with cities using mapbox.

        Args:
            df (dataframe): dataframe with mapping info.
            hover_data (list, optional): list of params to show on hover.
            file_name (str, optional): name of file
        """
        # Draw map
        fig = px.scatter_map(df,
                             lat="lat",
                             lon="lon",
                             hover_data=hover_data,
                             hover_name="city",
                             color=df["continent"],
                             zoom=1.3)  # type: ignore
        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  # Reduce margins
            # modebar_remove=["toImage"],  # remove modebar image button
            # showlegend=False,  # hide legend if not needed
            # annotations=[],  # remove any extra annotations
            mapbox=dict(zoom=1.3),
            font=dict(family=common.get_configs('font_family'),  # update font family
                      size=common.get_configs('font_size'))  # update font size
        )
        # Save and display the figure
        plots_class.save_plotly_figure(fig, file_name, save_final=True)

    @staticmethod
    def pedestrian_crossing(dataframe, min_x, max_x, person_id):
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
        crossed_ids = dataframe[(dataframe["YOLO_id"] == person_id)]

        # Group entries by Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

        # Filter entries based on x-coordinate boundaries
        filtered_crossed_ids = crossed_ids_grouped.filter(
            lambda x: (x["X-center"] <= min_x).any() and (x["X-center"] >= max_x).any())

        # Get unique IDs of the person who crossed the road within boundaries
        crossed_ids = filtered_crossed_ids["Unique Id"].unique()
        return crossed_ids

    @staticmethod
    def calculate_cell_phones(df_mapping, dfs):
        """Plots the relationship between average cell phone usage per person detected vs. traffic mortality.

        Args:
            df_mapping (DataFrame): DataFrame containing mapping information.
            dfs (dict): Dictionary of DataFrames containing video data.
        """
        info, no_person, total_time = {}, {}, {}
        time_ = []
        for key, value in tqdm(dfs.items(), total=len(dfs)):
            # Extract relevant information using the find_values function
            result = values_class.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                # Unpack the result since it's not None
                start = result[1]
                end = result[2]
                condition = result[3]
                city = result[4]
                lat = result[6]
                long = result[7]

                city_id_format = f'{city}_{lat}_{long}_{condition}'

                # Count the number of mobile objects in the video
                mobile_ids = Analysis.count_object(value, 67)

                # Calculate the duration of the video
                duration = end - start
                time_.append(duration)

                # Count the number of people in the video
                num_person = Analysis.count_object(value, 0)

                # Calculate average cell phones detected per person
                if num_person == 0 or mobile_ids == 0:
                    continue

                # Update the information dictionary
                if city_id_format in info:
                    previous_value = info[city_id_format]
                    # Extracting the old number of detected mobiles
                    previous_value = previous_value * no_person[city_id_format] * total_time[city_id_format] / 1000 / 60  # noqa: E501

                    # Summing up the previous value and the new value
                    total_value = previous_value + mobile_ids
                    no_person[city_id_format] += num_person
                    total_time[city_id_format] += duration

                    # Normalising with respect to total person detected and time
                    info[city_id_format] = (((total_value * 60) / total_time[city_id_format]) / no_person[city_id_format]) * 1000  # noqa: E501
                    continue  # Skip saving the variable in plotting variables
                else:
                    no_person[city_id_format] = num_person
                    total_time[city_id_format] = duration

                    """Normalising the detection with respect to time and numvber of person in the video.
                    Multiplied by 1000 to increase the value to look better in plotting."""

                    avg_cell_phone = (((mobile_ids * 60) / time_[-1]) / num_person) * 1000
                    info[city_id_format] = avg_cell_phone

            else:
                # Handle the case where no data was found for the given key
                logger.error(f"No matching data found for key: {key}")

        return info

    def calculate_traffic(self, df_mapping, dfs, normalised=True, person=0, bicycle=0,
                          motorcycle=0, car=0, bus=0, truck=0):
        """Plots the relationship between vehicle detection and crossing time.

        Args:
            df_mapping (DataFrame): DataFrame containing mapping information.
            dfs (dict): Dictionary of DataFrames containing video data.
            data (dict): Dictionary containing information about which object is crossing.
            bicycle (int, optional): Flag to include bicycle. Default is 0.
            motorcycle (int, optional): Flag to include motorcycles. Default is 0.
            car (int, optional): Flag to include cars. Default is 0.
            bus (int, optional): Flag to include buses. Default is 0.
            truck (int, optional): Flag to include trucks. Default is 0.
        """

        info, layer = {}, {}
        time_ = []

        # Iterate through each video DataFrame
        for key, value in tqdm(dfs.items(), total=len(dfs)):
            result = values_class.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                # Unpack the result since it's not None
                start = result[1]
                end = result[2]

                # Calculate the duration of the video
                duration = end - start
                time_.append(duration)

                dataframe = value

                # Filter vehicles based on flags
                if motorcycle == 1 & car == 1 & bus == 1 & truck == 1:
                    vehicle_ids = dataframe[(dataframe["YOLO_id"] == 2) | (dataframe["YOLO_id"] == 3) |
                                            (dataframe["YOLO_id"] == 5) | (dataframe["YOLO_id"] == 7)]

                elif motorcycle == 1:
                    vehicle_ids = dataframe[(dataframe["YOLO_id"] == 2)]

                elif car == 1:
                    vehicle_ids = dataframe[(dataframe["YOLO_id"] == 3)]

                elif bus == 1:
                    vehicle_ids = dataframe[(dataframe["YOLO_id"] == 5)]

                elif truck == 1:
                    vehicle_ids = dataframe[(dataframe["YOLO_id"] == 7)]

                elif bicycle == 1:
                    vehicle_ids = dataframe[(dataframe["YOLO_id"] == 1)]

                elif person == 1:
                    vehicle_ids = dataframe[(dataframe["YOLO_id"] == 0)]

                else:
                    logger.info("No plot generated")

                vehicle_ids = vehicle_ids["Unique Id"].unique()

                if vehicle_ids is None:
                    continue

                if normalised:
                    # Calculate normalised vehicle detection rate
                    new_value = ((len(vehicle_ids)/time_[-1]) * 60)
                else:
                    new_value = len(vehicle_ids)

            layer[key] = new_value
        info = wrapper_class.city_country_wrapper(input_dict=layer, mapping=df_mapping)

        return info

    def calculate_traffic_signs(self, df_mapping, dfs, normalised=True):
        """Plots traffic safety vs traffic mortality.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
        """
        info, layer = {}, {}  # Dictionaries to store information and duration

        # Loop through each video data
        for key, value in tqdm(dfs.items(), total=len(dfs)):

            # Extract relevant information using the find_values function
            result = values_class.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                start = result[1]
                end = result[2]

                dataframe = value
                duration = end - start

                # Filter dataframe for traffic instruments (YOLO_id 9 and 11)
                instrument = dataframe[(dataframe["YOLO_id"] == 9) | (dataframe["YOLO_id"] == 11)]

                instrument_ids = instrument["Unique Id"].unique()

                # Skip if there are no instrument ids
                if instrument_ids is None:
                    continue

                if normalised:
                    # Calculate count of traffic instruments detected per minute
                    count_ = ((len(instrument_ids)/duration) * 60)
                else:
                    count_ = len(instrument_ids)

            layer[key] = count_
        info = wrapper_class.city_country_wrapper(input_dict=layer, mapping=df_mapping)

        return info

    @staticmethod
    def crossing_event_wt_traffic_equipment(df_mapping, dfs, data):
        """Crossing events with respect to traffic equipment.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
            data (dict): Dictionary containing pedestrian crossing data.
        """
        time_ = {}
        counter_1, counter_2 = {}, {}

        # For a specific id of a person search for the first and last occurrence of that id and see if the traffic
        # light was present between it or not. Only getting those unique_id of the person who crosses the road.

        # Loop through each video data
        for key, df in tqdm(data.items(), total=len(data)):

            counter_exists, counter_nt_exists = 0, 0

            # Extract relevant information using the find_values function
            result = values_class.find_values_with_video_id(df_mapping, key)

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
                if f'{city}_{lat}_{long}_{condition}' in time_:
                    time_[f'{city}_{lat}_{long}_{condition}'] += duration
                else:
                    time_[f'{city}_{lat}_{long}_{condition}'] = duration

                value = dfs.get(key)

                for id, time in df.items():
                    unique_id_indices = value.index[value['Unique Id'] == id]
                    first_occurrence = unique_id_indices[0]
                    last_occurrence = unique_id_indices[-1]

                    # Check if YOLO_id = 9 and 11 exists within the specified index range
                    yolo_id_9_exists = any(
                        value.loc[first_occurrence:last_occurrence, 'YOLO_id'].isin([9, 11]))
                    yolo_id_9_not_exists = not any(
                        value.loc[first_occurrence:last_occurrence, 'YOLO_id'].isin([9, 11]))

                    if yolo_id_9_exists:
                        counter_exists += 1
                    if yolo_id_9_not_exists:
                        counter_nt_exists += 1

                counter_1[f'{city}_{lat}_{long}_{condition}'] = counter_1.get(f'{city}_{lat}_{long}_{condition}',
                                                                              0) + counter_exists
                counter_2[f'{city}_{lat}_{long}_{condition}'] = counter_2.get(f'{city}_{lat}_{long}_{condition}',
                                                                              0) + counter_nt_exists
        return counter_1, counter_2, time_

    # TODO: combine methods for looking at crossing events with/without traffic lights
    @staticmethod
    def crossing_event_wt_traffic_light(df_mapping, dfs, data):
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
            result = values_class.find_values_with_video_id(df_mapping, key)

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

                value = dfs.get(key)

                # Extract the time of day
                condition = time_of_day

                for id, time in df.items():
                    unique_id_indices = value.index[value['Unique Id'] == id]
                    first_occurrence = unique_id_indices[0]
                    last_occurrence = unique_id_indices[-1]

                    # Check if YOLO_id = 9 exists within the specified index range
                    yolo_id_9_exists = any(
                        value.loc[first_occurrence:last_occurrence, 'YOLO_id'] == 9)
                    yolo_id_9_not_exists = not any(
                        value.loc[first_occurrence:last_occurrence, 'YOLO_id'] == 9)

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

    @staticmethod
    def pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping):
        final = {}
        count = {key: len(value['ids']) for key, value in pedestrian_crossing_count.items()}

        for key, df in count.items():
            result = values_class.find_values_with_video_id(df_mapping, key)

            if result is not None:
                condition = result[3]
                city = result[4]
                lat = result[6]
                long = result[7]

                # Create the city_time_key (city + time_of_day)
                city_time_key = f'{city}_{lat}_{long}_{condition}'

                # Add the count to the corresponding city_time_key in the final dict
                if city_time_key in final:
                    final[city_time_key] += count[key]  # Add the current count to the existing sum
                else:
                    final[city_time_key] = count[key]

        return final

    # Plotting functions:
    @staticmethod
    def speed_and_time_to_start_cross(df_mapping, font_size_captions=40, x_axis_title_height=150, legend_x=0.81,
                                      legend_y=0.98, legend_spacing=0.02):
        logger.info("Plotting speed_and_time_to_start_cross")
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[25]
        avg_time = data_tuple[24]

        # Check if both 'speed' and 'time' are valid dictionaries
        if avg_speed is None or avg_time is None:
            raise ValueError("Either 'speed' or 'time' returned None, please check the input data or calculations.")

        # Remove the ones where there is data missing for a specific country and condition
        common_keys = avg_speed.keys() & avg_time.keys()

        # Retain only the key-value pairs where the key is present in both dictionaries
        avg_speed = {key: avg_speed[key] for key in common_keys}
        avg_time = {key: avg_time[key] for key in common_keys}

        # Now populate the final_dict with city-wise data
        for city_condition, speed in tqdm(avg_speed.items()):
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:

                # Initialise the city's dictionary if not already present
                if f'{city}_{lat}_{long}' not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {
                        "speed_0": None, "speed_1": None, "time_0": None, "time_1": None,
                        "country": country, "iso": iso_code}

                # Populate the corresponding speed and time based on the condition
                final_dict[f"{city}_{lat}_{long}"][f"speed_{condition}"] = speed
                if f'{city}_{lat}_{long}_{condition}' in avg_time:
                    final_dict[f"{city}_{lat}_{long}"][f"time_{condition}"] = avg_time[f'{city}_{lat}_{long}_{condition}']  # noqa: E501

        # Extract all valid speed_0 and speed_1 values along with their corresponding cities
        diff_speed_values = [(f'{city}', abs(data['speed_0'] - data['speed_1']))
                             for city, data in final_dict.items()
                             if data['speed_0'] is not None and data['speed_1'] is not None]

        if diff_speed_values:
            # Sort the list by the absolute difference and get the top 5 and bottom 5
            sorted_diff_speed_values = sorted(diff_speed_values, key=lambda x: x[1], reverse=True)

            top_5_max_speed = sorted_diff_speed_values[:5]  # Top 5 maximum differences
            top_5_min_speed = sorted_diff_speed_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("Top 5 cities with max |speed at day - speed at night| differences:")
            for city, diff in top_5_max_speed:
                city_state = wrapper_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")

            logger.info("Top 5 cities with min |speed at day - speed at night| differences:")
            for city, diff in top_5_min_speed:
                city_state = wrapper_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")
        else:
            logger.info("No valid speed_0 and speed_1 values found for comparison.")

        # Extract all valid time_0 and time_1 values along with their corresponding cities
        diff_time_values = [(city, abs(data['time_0'] - data['time_1']))
                            for city, data in final_dict.items()
                            if data['time_0'] is not None and data['time_1'] is not None]

        if diff_time_values:
            sorted_diff_time_values = sorted(diff_time_values, key=lambda x: x[1], reverse=True)

            top_5_max = sorted_diff_time_values[:5]  # Top 5 maximum differences
            top_5_min = sorted_diff_time_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("Top 5 cities with max |time_0 - time_1| differences:")
            for city, diff in top_5_max:
                city_state = wrapper_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")

            logger.info("Top 5 cities with min |time_0 - time_1| differences:")
            for city, diff in top_5_min:
                city_state = wrapper_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")
        else:
            logger.info("No valid time_0 and time_1 values found for comparison.")

        # Filtering out entries where entries is None
        filtered_dict_s_0 = {city: info for city, info in final_dict.items() if info["speed_0"] is not None}
        filtered_dict_s_1 = {city: info for city, info in final_dict.items() if info["speed_1"] is not None}
        filtered_dict_t_0 = {city: info for city, info in final_dict.items() if info["time_0"] is not None}
        filtered_dict_t_1 = {city: info for city, info in final_dict.items() if info["time_1"] is not None}

        # Find city with max and min speed_0 and speed_1
        if filtered_dict_s_0:
            max_speed_city_0 = max(filtered_dict_s_0, key=lambda city: filtered_dict_s_0[city]["speed_0"])
            min_speed_city_0 = min(filtered_dict_s_0, key=lambda city: filtered_dict_s_0[city]["speed_0"])
            max_speed_value_0 = filtered_dict_s_0[max_speed_city_0]["speed_0"]
            min_speed_value_0 = filtered_dict_s_0[min_speed_city_0]["speed_0"]

            logger.info(f"City with max speed at day: {wrapper_class.process_city_string(max_speed_city_0, df_mapping)} with speed of {max_speed_value_0} m/s")  # noqa:E501
            logger.info(f"City with min speed at day: {wrapper_class.process_city_string(min_speed_city_0, df_mapping)} with speed of {min_speed_value_0} m/s")  # noqa:E501

        if filtered_dict_s_1:
            max_speed_city_1 = max(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            min_speed_city_1 = min(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            max_speed_value_1 = filtered_dict_s_1[max_speed_city_1]["speed_1"]
            min_speed_value_1 = filtered_dict_s_1[min_speed_city_1]["speed_1"]

            logger.info(f"City with max speed at night: {wrapper_class.process_city_string(max_speed_city_1, df_mapping)} with speed of {max_speed_value_1} m/s")  # noqa:E501
            logger.info(f"City with min speed at night: {wrapper_class.process_city_string(min_speed_city_1, df_mapping)} with speed of {min_speed_value_1} m/s")  # noqa:E501

        # Find city with max and min time_0 and time_1
        if filtered_dict_t_0:
            max_time_city_0 = max(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            min_time_city_0 = min(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            max_time_value_0 = filtered_dict_t_0[max_time_city_0]["time_0"]
            min_time_value_0 = filtered_dict_t_0[min_time_city_0]["time_0"]

            logger.info(f"City with max time at day: {wrapper_class.process_city_string(max_time_city_0, df_mapping)} with time of {max_time_value_0} s")  # noqa:E501
            logger.info(f"City with min time at day: {wrapper_class.process_city_string(min_time_city_0, df_mapping)} with time of {min_time_value_0} s")  # noqa:E501

        if filtered_dict_t_1:
            max_time_city_1 = max(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            min_time_city_1 = min(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            max_time_value_1 = filtered_dict_t_1[max_time_city_1]["time_1"]
            min_time_value_1 = filtered_dict_t_1[min_time_city_1]["time_1"]

            logger.info(f"City with max time at night: {wrapper_class.process_city_string(max_time_city_1, df_mapping)} with time of {max_time_value_1} s")  # noqa:E501
            logger.info(f"City with min time at night: {wrapper_class.process_city_string(min_time_city_1, df_mapping)} with time of {min_time_value_1} s")  # noqa:E501

        # Extract valid speed and time values and calculate statistics
        speed_0_values = [data['speed_0'] for data in final_dict.values() if pd.notna(data['speed_0'])]
        speed_1_values = [data['speed_1'] for data in final_dict.values() if pd.notna(data['speed_1'])]
        time_0_values = [data['time_0'] for data in final_dict.values() if pd.notna(data['time_0'])]
        time_1_values = [data['time_1'] for data in final_dict.values() if pd.notna(data['time_1'])]

        if speed_0_values:
            mean_speed_0 = statistics.mean(speed_0_values)
            sd_speed_0 = statistics.stdev(speed_0_values) if len(speed_0_values) > 1 else 0
            logger.info(f"Mean of speed during day time: {mean_speed_0}")
            logger.info(f"Standard deviation of speed during day time: {sd_speed_0}")
        else:
            logger.error("No valid speed during day time values found.")

        if speed_1_values:
            mean_speed_1 = statistics.mean(speed_1_values)
            sd_speed_1 = statistics.stdev(speed_1_values) if len(speed_1_values) > 1 else 0
            logger.info(f"Mean of speed during night time: {mean_speed_1}")
            logger.info(f"Standard deviation of speed during night time: {sd_speed_1}")
        else:
            logger.error("No valid speed during night time values found.")

        if time_0_values:
            mean_time_0 = statistics.mean(time_0_values)
            sd_time_0 = statistics.stdev(time_0_values) if len(time_0_values) > 1 else 0
            logger.info(f"Mean of time during day time: {mean_time_0}")
            logger.info(f"Standard deviation of time during day time: {sd_time_0}")
        else:
            logger.error("No valid time during day time values found.")

        if time_1_values:
            mean_time_1 = statistics.mean(time_1_values)
            sd_time_1 = statistics.stdev(time_1_values) if len(time_1_values) > 1 else 0
            logger.info(f"Mean of time during night time: {mean_time_1}")
            logger.info(f"Standard deviation of time during night time: {sd_time_1}")
        else:
            logger.error("No valid time during night time values found.")

        # Extract city, condition, and count_ from the info dictionary
        cities, conditions_, counts = [], [], []
        for key, value in tqdm(avg_time.items()):
            city, lat, long, condition = key.split('_')
            cities.append(f'{city}_{lat}_{long}')
            conditions_.append(condition)
            counts.append(value)

        # Combine keys from speed and time to ensure we include all available cities and conditions
        all_keys = set(avg_speed.keys()).union(set(avg_time.keys()))

        # Extract unique cities
        cities = list(set(["_".join(key.split('_')[:2]) for key in all_keys]))

        country_city_map = {}
        for city_state, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city_state)

        # Flatten the city list based on country groupings
        cities_ordered = []
        for country in sorted(country_city_map.keys()):  # Sort countries alphabetically
            cities_in_country = sorted(country_city_map[country])  # Sort cities within each country alphabetically
            cities_ordered.extend(cities_in_country)

        # Prepare data for day and night stacking
        day_avg_speed = [final_dict[city]['speed_0'] for city in cities_ordered]
        night_avg_speed = [final_dict[city]['speed_1'] for city in cities_ordered]
        day_time_dict = [final_dict[city]['time_0'] for city in cities_ordered]
        night_time_dict = [final_dict[city]['time_1'] for city in cities_ordered]

        # Ensure that plotting uses cities_ordered
        assert len(cities_ordered) == len(day_avg_speed) == len(night_avg_speed) == len(
            day_time_dict) == len(night_time_dict), "Lengths of lists don't match!"

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups

        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col * 2, cols=2,  # Two columns
            vertical_spacing=0,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[2.0] * (num_cities_per_col * 2),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city = wrapper_class.process_city_string(city, df_mapping)

            # Row for speed (Day and Night)
            row = 2 * i + 1
            if day_avg_speed[i] is not None and night_avg_speed[i] is not None:
                value = (day_avg_speed[i] + night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_avg_speed[i] is not None:  # Only day data available
                value = (day_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_avg_speed[i] is not None:  # Only night data available
                value = (night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night",
                    marker=dict(color=bar_colour_2), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            # Row for time (Day and Night)
            row = 2 * i + 2
            if day_time_dict[i] is not None and night_time_dict[i] is not None:
                value = (day_time_dict[i] + night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_4), text=[''],
                    textposition='auto', showlegend=False), row=row, col=1)

            elif day_time_dict[i] is not None:  # Only day time data available
                value = (day_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_time_dict[i] is not None:  # Only night time data available
                value = (night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_4),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

        # Similarly for the right column
        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city = wrapper_class.process_city_string(city, df_mapping)

            row = 2 * i + 1
            idx = num_cities_per_col + i
            if day_avg_speed[idx] is not None and night_avg_speed[idx] is not None:
                value = (day_avg_speed[idx] + night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_avg_speed[idx] is not None:
                value = (day_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_avg_speed[idx] is not None:
                value = (night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            row = 2 * i + 2
            if day_time_dict[idx] is not None and night_time_dict[idx] is not None:
                value = (day_time_dict[idx] + night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_4), text=[''],
                    textposition='inside', showlegend=False), row=row, col=2)

            elif day_time_dict[idx] is not None:  # Only day time data available
                value = (day_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_time_dict[idx] is not None:  # Only night time data available
                value = (night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_4),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_time = max([
            (day_time_dict[i] if day_time_dict[i] is not None else 0) +
            (night_time_dict[i] if night_time_dict[i] is not None else 0)
            for i in range(len(cities_ordered))
        ]) if cities_ordered else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=False
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=False
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(
            title=dict(text="Mean speed of crossing (in m/s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=1
        )
        fig.update_xaxes(
            title=dict(text="Mean speed of crossing (in m/s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=2
        )
        fig.update_xaxes(
            title=dict(text="Mean time to start crossing (in s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=num_cities_per_col * 2,
            col=1
        )

        fig.update_xaxes(
            title=dict(text="Mean time to start crossing (in s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=num_cities_per_col * 2,
            col=2
        )

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=TALL_FIG_HEIGHT*2, width=4960, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Set the x-axis range to cover the values you want in x_grid_values
        x_grid_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x', yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x2', yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Define the legend items
        legend_items = [
            {"name": "Crossing speed during daytime", "color": bar_colour_1},
            {"name": "Crossing speed during night time", "color": bar_colour_2},
            {"name": "Crossing decision time during daytime", "color": bar_colour_3},
            {"name": "Crossing decision time during night time", "color": bar_colour_4},
        ]

        # Add the vertical legends at the top and bottom
        plots_class.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
                                                    spacing=legend_spacing, font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=1, x1=0.495, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.505, y0=1, x1=1, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Split cities into left and right columns
        left_column_cities = cities_ordered[:num_cities_per_col]
        right_column_cities = cities_ordered[num_cities_per_col:]

        # Adjust x positioning for the left and right columns
        x_position_left = 0.0  # Position for the left column
        x_position_right = 1.0  # Position for the right column
        font_size = 15  # Font size for visibility

        # Initialize variables for dynamic y positioning for both columns
        current_row_left = 1  # Start from the first row for the left column
        current_row_right = 1  # Start from the first row for the right column
        y_position_map_left = {}  # Store y positions for each country (left column)
        y_position_map_right = {}  # Store y positions for each country (right column)

        # Calculate the y positions dynamically for the left column
        for city in left_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_left:  # Add the country label once per country
                y_position_map_left[country] = 1 - (current_row_left - 1) / ((len(left_column_cities)-0.56) * 2)

            current_row_left += 2  # Increment the row for each city (speed and time take two rows)

        # Calculate the y positions dynamically for the right column
        for city in right_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_right:  # Add the country label once per country
                y_position_map_right[country] = 1 - (current_row_right - 1) / ((len(right_column_cities)-0.56) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        # Add annotations for country names dynamically for the left column
        for country, y_position in y_position_map_left.items():
            iso2 = wrapper_class.iso3_to_iso2(country)
            country = country + wrapper_class.iso2_to_flag(iso2)
            fig.add_annotation(
                x=x_position_left,  # Left column x position
                y=y_position,  # Calculated y position based on the city order
                xref="paper", yref="paper",
                text=country,  # Country name
                showarrow=False,
                font=dict(size=font_size, color="black"),
                xanchor='right',
                align='right',
                bgcolor='rgba(255,255,255,0.8)',  # Background color for visibility
                # bordercolor="black",  # Border for visibility
            )

        # Add annotations for country names dynamically for the right column
        for country, y_position in y_position_map_right.items():
            iso2 = wrapper_class.iso3_to_iso2(country)
            country = country + wrapper_class.iso2_to_flag(iso2)
            fig.add_annotation(
                x=x_position_right,  # Right column x position
                y=y_position,  # Calculated y position based on the city order
                xref="paper", yref="paper",
                text=country,  # Country name
                showarrow=False,
                font=dict(size=font_size, color="black"),
                xanchor='left',
                align='left',
                bgcolor='rgba(255,255,255,0.8)',  # Background color for visibility
                # bordercolor="black",  # Border for visibility
            )

        fig.update_yaxes(
            tickfont=dict(size=14, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )

        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=80, t=x_axis_title_height, b=x_axis_title_height))

        plots_class.save_plotly_figure(fig,
                                       "consolidated",
                                       height=TALL_FIG_HEIGHT*2,
                                       width=4960,
                                       scale=SCALE,
                                       save_final=True,
                                       save_eps=False,
                                       save_png=False)

    @staticmethod
    def safe_average(values):
        # Filter out None and NaN values.
        valid_values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(valid_values) / len(valid_values) if valid_values else 0

    @staticmethod
    def plot_crossing_without_traffic_light(df_mapping, font_size_captions=40, x_axis_title_height=150, legend_x=0.92,
                                            legend_y=0.015, legend_spacing=0.02):
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        without_trf_light = data_tuple[28]
        # Now populate the final_dict with city-wise speed data
        for city_condition, count in without_trf_light.items():
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:
                # Initialise the city's dictionary if not already present
                if f"{city}_{lat}_{long}" not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {"without_trf_light_0": None, "without_trf_light_1": None,
                                                          "country": country, "iso": iso_code}

                # normalise by total time and total number of detected persons
                total_time = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "total_time")
                person = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "person")
                count = count / total_time / person

                # Populate the corresponding speed based on the condition
                final_dict[f"{city}_{lat}_{long}"][f"without_trf_light_{condition}"] = count

        # Multiply each of the numeric speed values by 10^6
        for city_key, data in final_dict.items():
            for key, value in data.items():
                # Only modify keys that represent speed values
                if key.startswith("without_trf_light") and value is not None:
                    data[key] = round(value * 10**6, 2)

        cities_ordered = sorted(
            final_dict.keys(),
            key=lambda city: Analysis.safe_average([
                final_dict[city]["without_trf_light_0"],
                final_dict[city]["without_trf_light_1"]
            ]),
            reverse=True
        )

        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        # Prepare data for day and night stacking
        day_crossing = [final_dict[city]['without_trf_light_0'] for city in cities_ordered]
        night_crossing = [final_dict[city]['without_trf_light_1'] for city in cities_ordered]

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups
        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            if day_crossing[i] is not None and night_crossing[i] is not None:
                value = round((day_crossing[i] + night_crossing[i])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=1)

            elif day_crossing[i] is not None:  # Only day data available
                value = (day_crossing[i])
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_crossing[i] is not None:  # Only night data available
                value = (night_crossing[i])
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            idx = num_cities_per_col + i
            if day_crossing[idx] is not None and night_crossing[idx] is not None:
                value = round((day_crossing[idx] + night_crossing[idx])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_crossing[idx] is not None:
                value = (day_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_crossing[idx] is not None:
                value = (night_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_crossing[i] if day_crossing[i] is not None else 0) +
            (night_crossing[i] if night_crossing[i] is not None else 0)
            for i in range(len(cities))
        ]) if cities else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=True
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=True
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(title=dict(text="Road crossings without traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=1)

        fig.update_xaxes(title=dict(text="Road crossings without traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=2)

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=TALL_FIG_HEIGHT, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Manually add gridlines using `shapes`
        x_grid_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600]  # Define the gridline positions on the x-axis

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x', yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x2', yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Define the legend items
        legend_items = [
            {"name": "Day", "color": bar_colour_1},
            {"name": "Night", "color": bar_colour_2},
        ]

        # Add the vertical legends at the top and bottom
        plots_class.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
                                                    spacing=legend_spacing, font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=1, x1=0.495, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.505, y0=1, x1=1, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Split cities into left and right columns
        left_column_cities = cities_ordered[:num_cities_per_col]
        right_column_cities = cities_ordered[num_cities_per_col:]

        # Initialize variables for dynamic y positioning for both columns
        current_row_left = 1  # Start from the first row for the left column
        current_row_right = 1  # Start from the first row for the right column
        y_position_map_left = {}  # Store y positions for each country (left column)
        y_position_map_right = {}  # Store y positions for each country (right column)

        # Calculate the y positions dynamically for the left column
        for city in left_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_left:  # Add the country label once per country
                y_position_map_left[country] = 1 - (current_row_left - 1) / (len(left_column_cities) * 2)

            current_row_left += 2  # Increment the row for each city (speed and time take two rows)

        # Calculate the y positions dynamically for the right column
        for city in right_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_right:  # Add the country label once per country
                y_position_map_right[country] = 1 - (current_row_right - 1) / (len(right_column_cities) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        fig.update_yaxes(
            tickfont=dict(size=12, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=x_axis_title_height, b=180))
        plots_class.save_plotly_figure(fig,
                                       "crossings_without_traffic_equipment_avg",
                                       width=2480,
                                       height=TALL_FIG_HEIGHT,
                                       scale=SCALE,
                                       save_eps=False,
                                       save_final=True)

    @staticmethod
    def plot_crossing_with_traffic_light(df_mapping, font_size_captions=40, x_axis_title_height=150, legend_x=0.92,
                                         legend_y=0.015, legend_spacing=0.02):
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        with_trf_light = data_tuple[27]
        # Now populate the final_dict with city-wise speed data
        for city_condition, count in with_trf_light.items():
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:
                # Initialize the city's dictionary if not already present
                if f"{city}_{lat}_{long}" not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {"with_trf_light_0": None, "with_trf_light_1": None,
                                                          "country": country, "iso": iso_code}

                # normalise by total time and total number of detected persons
                total_time = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "total_time")
                person = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "person")
                count = count / total_time / person

                # Populate the corresponding speed based on the condition
                final_dict[f"{city}_{lat}_{long}"][f"with_trf_light_{condition}"] = count

        # Multiply each of the numeric speed values by 10^6
        for city_key, data in final_dict.items():
            for key, value in data.items():
                # Only modify keys that represent speed values
                if key.startswith("with_trf_light") and value is not None:
                    data[key] = round(value * 10**6, 2)

        cities_ordered = sorted(
            final_dict.keys(),
            key=lambda city: Analysis.safe_average([
                final_dict[city]["with_trf_light_0"],
                final_dict[city]["with_trf_light_1"]
            ]),
            reverse=True
        )

        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        # Prepare data for day and night stacking
        day_crossing = [final_dict[city]['with_trf_light_0'] for city in cities_ordered]
        night_crossing = [final_dict[city]['with_trf_light_1'] for city in cities_ordered]

        # # Ensure that plotting uses cities_ordered
        # assert len(cities_ordered) == len(day_crossing) == len(night_crossing), "Lengths of lists don't match!"

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups
        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            if day_crossing[i] is not None and night_crossing[i] is not None:
                value = round((day_crossing[i] + night_crossing[i])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_crossing[i] is not None:  # Only day data available
                value = (day_crossing[i])
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_crossing[i] is not None:  # Only night data available
                value = (night_crossing[i])
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            idx = num_cities_per_col + i
            if day_crossing[idx] is not None and night_crossing[idx] is not None:
                value = round((day_crossing[idx] + night_crossing[idx])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_crossing[idx] is not None:
                value = (day_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_crossing[idx] is not None:
                value = (night_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_crossing[i] if day_crossing[i] is not None else 0) +
            (night_crossing[i] if night_crossing[i] is not None else 0)
            for i in range(len(cities))
        ]) if cities else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=True
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=True
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(title=dict(text="Road crossings with traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=1)

        fig.update_xaxes(title=dict(text="Road crossings with traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=2)

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=TALL_FIG_HEIGHT, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Manually add gridlines using `shapes`
        x_grid_values = [50, 100, 150, 200, 250]  # Define the gridline positions on the x-axis

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x', yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x2', yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Define the legend items
        legend_items = [
            {"name": "Day", "color": bar_colour_1},
            {"name": "Night", "color": bar_colour_2},
        ]

        # Add the vertical legends at the top and bottom
        plots_class.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
                                                    spacing=legend_spacing, font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=1, x1=0.495, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.505, y0=1, x1=1, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Split cities into left and right columns
        left_column_cities = cities_ordered[:num_cities_per_col]
        right_column_cities = cities_ordered[num_cities_per_col:]

        # Initialize variables for dynamic y positioning for both columns
        current_row_left = 1  # Start from the first row for the left column
        current_row_right = 1  # Start from the first row for the right column
        y_position_map_left = {}  # Store y positions for each country (left column)
        y_position_map_right = {}  # Store y positions for each country (right column)

        # Calculate the y positions dynamically for the left column
        for city in left_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_left:  # Add the country label once per country
                y_position_map_left[country] = 1 - (current_row_left - 1) / (len(left_column_cities) * 2)

            current_row_left += 2  # Increment the row for each city (speed and time take two rows)

        # Calculate the y positions dynamically for the right column
        for city in right_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_right:  # Add the country label once per country
                y_position_map_right[country] = 1 - (current_row_right - 1) / (len(right_column_cities) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        fig.update_yaxes(
            tickfont=dict(size=12, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=x_axis_title_height, b=180))
        plots_class.save_plotly_figure(fig,
                                       "crossings_with_traffic_equipment_avg",
                                       width=2480,
                                       height=TALL_FIG_HEIGHT,
                                       scale=SCALE,
                                       save_eps=False,
                                       save_final=True)

    def compute_avg_variable_city(self, variable_city):
        """
        Compute the average value for each city-condition key in a nested dictionary.
        """
        avg_dict = {}

        for key, inner_dict in variable_city.items():
            # Compute average
            values = list(inner_dict.values())
            avg_value = sum(values) / len(values) if values else 0

            # Assign average directly to the same key
            avg_dict[key] = avg_value

        return avg_dict

    @staticmethod
    def correlation_matrix(df_mapping):
        """
        Compute and visualize correlation matrices for various city-level traffic and demographic data.

        This method:
        - Loads precomputed statistical data from a pickled file.
        - Aggregates metrics like speed, time, vehicle/pedestrian counts, and socioeconomic indicators.
        - Constructs structured dictionaries for day (condition 0) and night (condition 1) conditions.
        - Computes Spearman correlation matrices for:
            - Daytime data
            - Nighttime data
            - Averaged across both conditions
            - Per continent basis
        - Generates and saves Plotly heatmaps for all computed correlation matrices.

        Args:
            df_mapping (pd.DataFrame): A mapping DataFrame containing metadata for each city-state combination,
                                       including country, continent, GDP, literacy rate, etc.

        Raises:
            ValueError: If essential data (e.g., average speed or time) is missing.
        """
        logger.info("Plotting correlation matrices.")
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        (ped_cross_city, ped_crossing_count, person_city, bicycle_city, car_city,
         motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city,
         cellphone_city, trf_sign_city, speed_values, time_values, avg_time, avg_speed) = data_tuple[10:26]

        # Check if both 'speed' and 'time' are valid dictionaries
        if avg_speed is None or avg_time is None:
            raise ValueError("Either 'speed' or 'time' returned None, please check the input data or calculations.")

        # Remove the ones where there is data missing for a specific country and condition
        common_keys = avg_speed.keys() & avg_time.keys()

        # Retain only the key-value pairs where the key is present in both dictionaries
        avg_speed = {key: avg_speed[key] for key in common_keys}
        avg_time = {key: avg_time[key] for key in common_keys}

        # Now populate the final_dict with city-wise data
        for city_condition, speed in avg_speed.items():
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            continent = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "continent")
            population_country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "population_country")  # noqa: E501
            gdp_city = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "gmp")
            traffic_mortality = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "traffic_mortality")  # noqa: E501
            literacy_rate = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "literacy_rate")
            gini = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "gini")
            traffic_index = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "traffic_index")

            if country or iso_code is not None:

                # Initialize the city's dictionary if not already present
                if f'{city}_{lat}_{long}' not in final_dict:
                    final_dict[f'{city}_{lat}_{long}'] = {
                        "avg_speed_0": None, "avg_speed_1": None, "avg_time_0": None, "avg_time_1": None,
                        "speed_val_0": None, "speed_val_1": None, "time_val_0": None, "time_val_1": None,
                        "ped_cross_city_0": 0, "ped_cross_city_1": 0,
                        "person_city_0": 0, "person_city_1": 0, "bicycle_city_0": 0,
                        "bicycle_city_1": 0, "car_city_0": 0, "car_city_1": 0,
                        "motorcycle_city_0": 0, "motorcycle_city_1": 0, "bus_city_0": 0,
                        "bus_city_1": 0, "truck_city_0": 0, "truck_city_1": 0,
                        "cross_evnt_city_0": 0, "cross_evnt_city_1": 0, "vehicle_city_0": 0,
                        "vehicle_city_1": 0, "cellphone_city_0": 0, "cellphone_city_1": 0,
                        "trf_sign_city_0": 0, "trf_sign_city_1": 0,
                    }

                # Populate the corresponding speed and time based on the condition
                final_dict[f'{city}_{lat}_{long}'][f"avg_speed_{condition}"] = speed
                if f'{city}_{lat}_{long}_{condition}' in avg_time:
                    final_dict[f'{city}_{lat}_{long}'][f"avg_time_{condition}"] = avg_time.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"time_val_{condition}"] = time_values.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"speed_val_{condition}"] = speed_values.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"time_val_{condition}"] = time_values.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"ped_cross_city_{condition}"] = ped_cross_city.get(
                        f'{city}_{lat}_{long}_{condition}', None)

                    avg_person_city = analysis_class.compute_avg_variable_city(person_city)
                    final_dict[f'{city}_{lat}_{long}'][f"person_city_{condition}"] = avg_person_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_bicycle_city = analysis_class.compute_avg_variable_city(bicycle_city)
                    final_dict[f'{city}_{lat}_{long}'][f"bicycle_city_{condition}"] = avg_bicycle_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_car_city = analysis_class.compute_avg_variable_city(car_city)
                    final_dict[f'{city}_{lat}_{long}'][f"car_city_{condition}"] = avg_car_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_motorcycle_city = analysis_class.compute_avg_variable_city(motorcycle_city)
                    final_dict[f'{city}_{lat}_{long}'][f"motorcycle_city_{condition}"] = avg_motorcycle_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_bus_city = analysis_class.compute_avg_variable_city(bus_city)
                    final_dict[f'{city}_{lat}_{long}'][f"bus_city_{condition}"] = avg_bus_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_truck_city = analysis_class.compute_avg_variable_city(truck_city)
                    final_dict[f'{city}_{lat}_{long}'][f"truck_city_{condition}"] = avg_truck_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    final_dict[f'{city}_{lat}_{long}'][f"cross_evnt_city_{condition}"] = cross_evnt_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_vehicle_city = analysis_class.compute_avg_variable_city(vehicle_city)
                    final_dict[f'{city}_{lat}_{long}'][f"vehicle_city_{condition}"] = avg_vehicle_city.get(
                        f'{city}_{lat}_{long}_{condition}', None)

                    final_dict[f'{city}_{lat}_{long}'][f"cellphone_city_{condition}"] = cellphone_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_trf_sign_city = analysis_class.compute_avg_variable_city(trf_sign_city)
                    final_dict[f'{city}_{lat}_{long}'][f"trf_sign_city_{condition}"] = avg_trf_sign_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    final_dict[f'{city}_{lat}_{long}'][f"traffic_mortality_{condition}"] = traffic_mortality
                    final_dict[f'{city}_{lat}_{long}'][f"literacy_rate_{condition}"] = literacy_rate
                    final_dict[f'{city}_{lat}_{long}'][f"gini_{condition}"] = gini
                    final_dict[f'{city}_{lat}_{long}'][f"traffic_index_{condition}"] = traffic_index
                    final_dict[f'{city}_{lat}_{long}'][f"continent_{condition}"] = continent
                    if gdp_city is not None:
                        final_dict[f'{city}_{lat}_{long}'][f"gmp_{condition}"] = gdp_city/population_country

        # Initialize an empty list to store the rows for the DataFrame
        data_day, data_night = [], []

        # Loop over each city and gather relevant values for condition 0
        for city in final_dict:
            # Initialize a dictionary for the row
            row_day, row_night = {}, {}

            # Add data for condition 0 (ignore 'speed_val' and 'time_val')
            for condition in ['0']:  # Only include condition 0
                for key, value in final_dict[city].items():
                    if condition in key and 'speed_val' not in key and 'time_val' not in key and 'continent' not in key:  # noqa:E501
                        row_day[key] = value

            # Append the row to the data list
            data_day.append(row_day)

            for condition in ['1']:  # Only include condition 1
                for key, value in final_dict[city].items():
                    if condition in key and 'speed_val' not in key and 'time_val' not in key and 'continent' not in key:  # noqa:E501
                        row_night[key] = value

            # Append the row to the data list
            data_night.append(row_night)

        # Convert the list of rows into a Pandas DataFrame
        df_day = pd.DataFrame(data_day)
        df_night = pd.DataFrame(data_night)

        # Calculate the correlation matrix
        corr_matrix_day = df_day.corr(method='spearman')
        corr_matrix_night = df_night.corr(method='spearman')

        # Rename the variables in the correlation matrix
        rename_dict_1 = {
            'avg_speed_0': 'Speed of', 'avg_speed_1': 'Crossing speed',
            'avg_time_0': 'Crossing decision time', 'avg_time_1': 'Crossing decision time',
            'ped_cross_city_0': 'Crossing', 'ped_cross_city_1': 'Crossing',
            'person_city_0': 'Detected persons', 'person_city_1': 'Detected persons',
            'bicycle_city_0': 'Detected bicycles', 'bicycle_city_1': 'Detected bicycles',
            'car_city_0': 'Detected cars', 'car_city_1': 'Detected cars',
            'motorcycle_city_0': 'Detected motorcycles', 'motorcycle_city_1': 'Detected motorcycles',
            'bus_city_0': 'Detected bus', 'bus_city_1': 'Detected bus',
            'truck_city_0': 'Detected truck', 'truck_city_1': 'Detected truck',
            'cross_evnt_city_0': 'Detected crossings without traffic lights',
            'cross_evnt_city_1': 'Detected crossings without traffic lights',
            'vehicle_city_0': 'Detected motor vehicles',
            'vehicle_city_1': 'Detected motor vehicles',
            'cellphone_city_0': 'Detected cellphones', 'cellphone_city_1': 'Detected cellphones',
            'trf_sign_city_0': 'Detected traffic signs', 'trf_sign_city_1': 'Detected traffic signs',
            'gmp_0': 'GMP', 'gmp_1': 'GMP',
            'traffic_mortality_0': 'Traffic mortality', 'traffic_mortality_1': 'Traffic mortality',
            'literacy_rate_0': 'Literacy rate', 'literacy_rate_1': 'Literacy rate',
            'gini_0': 'Gini coefficient', 'gini_1': 'Gini coefficient', 'traffic_index_0': 'Traffic index',
            'traffic_index_1': 'Traffic index'
            }

        corr_matrix_day = corr_matrix_day.rename(columns=rename_dict_1, index=rename_dict_1)
        corr_matrix_night = corr_matrix_night.rename(columns=rename_dict_1, index=rename_dict_1)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_day, text_auto=".2f",  # Display correlation values on the heatmap # type: ignore
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto")  # Automatically adjust aspect ratio
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        plots_class.save_plotly_figure(fig, "correlation_matrix_heatmap_day", save_final=True)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_night, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation Matrix Heatmap in night"  # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        plots_class.save_plotly_figure(fig, "correlation_matrix_heatmap_night", save_final=True)

        # Initialize a list to store rows of data (one row per city)
        data_rows = []

        # Assuming `conditions` is a list of conditions you are working with
        conditions = ['0', '1']  # Modify this list to include all conditions you have (e.g., '0', '1', etc.)

        # Iterate over each city and condition
        for city in final_dict:
            # Initialize a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'gmp', 'traffic_mortality', 'literacy_rate', 'gini', 'traffic_index']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[city].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[city].get(f"speed_val_{condition}", [])
                time_vals = final_dict[city].get(f"time_val_{condition}", [])

                if speed_vals:  # Avoid division by zero or empty dict
                    all_speed_values = [val for inner_dict in speed_vals.values() for val in inner_dict.values()]
                    if all_speed_values:  # Check to avoid computing mean on empty list
                        row_data[f"avg_speed_val_{condition}"] = np.mean(all_speed_values)
                    else:
                        row_data[f"avg_speed_val_{condition}"] = np.nan
                else:
                    row_data[f"avg_speed_val_{condition}"] = np.nan  # Handle empty or missing dict

                if time_vals:
                    all_time_values = [val for inner_dict in time_vals.values() for val in inner_dict.values()]
                    if all_time_values:
                        row_data[f"avg_time_val_{condition}"] = np.mean(all_time_values)
                    else:
                        row_data[f"avg_time_val_{condition}"] = np.nan
                else:
                    row_data[f"avg_time_val_{condition}"] = np.nan  # Handle empty or missing dict

            # Append the row data for the current city
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)

        # Create a new DataFrame to average the columns across conditions
        agg_df = pd.DataFrame()

        # Loop through the columns in the original DataFrame
        for col in df.columns:
            # Extract the feature name (without condition part)
            feature_name = "_".join(col.split("_")[:-1])
            condition = col.split("_")[-1]

            # Create a new column by averaging values across conditions for the same feature
            if feature_name not in agg_df.columns:
                # Select the columns for this feature across all conditions
                condition_cols = [c for c in df.columns if feature_name in c]
                agg_df[feature_name] = df[condition_cols].mean(axis=1)

        # Compute the correlation matrix on the aggregated DataFrame
        corr_matrix_avg = agg_df.corr(method='spearman')

        # Rename the variables in the correlation matrix (example: renaming keys)
        rename_dict_2 = {
            'avg_speed_val': 'Crossing speed', 'avg_time_val': 'Crossing decision time',
            'ped_cross_city': 'Crossing', 'person_city': 'Detected persons',
            'bicycle_city': 'Detected bicycles', 'car_city': 'Detected cars',
            'motorcycle_city': 'Detected motorcycles', 'bus_city': 'Detected bus',
            'truck_city': 'Detected truck', 'cross_evnt_city': 'Crossing without traffic light',
            'vehicle_city': 'Detected total number of motor vehicle', 'cellphone_city': 'Detected cellphone',
            'trf_sign_city': 'Detected traffic signs', 'gmp_city': 'GMP',
            'traffic_mortality_city': 'Traffic mortality', 'literacy_rate_city': 'Literacy rate',
            'gini': 'Gini coefficient', 'traffic_index': 'Traffic Index'
            }

        corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_2, index=rename_dict_2)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation matrix heatmap averaged" # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        plots_class.save_plotly_figure(fig, "correlation_matrix_heatmap_averaged", save_final=True)

        # Continent Wise

        # Initialise a list to store rows of data (one row per city)
        data_rows = []

        # Assuming `conditions` is a list of conditions you are working with
        conditions = ['0', '1']  # Modify this list to include all conditions you have (e.g., '0', '1', etc.)
        unique_continents = df_mapping['continent'].unique()

        # Iterate over each city and condition
        for city in final_dict:
            # Initialise a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'gmp', 'traffic_mortality', 'literacy_rate', 'continent', 'gini', 'traffic_index']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[city].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[city].get(f"speed_val_{condition}", [])
                time_vals = final_dict[city].get(f"time_val_{condition}", [])

                if speed_vals:
                    all_speed_values = [val for inner_dict in speed_vals.values() for val in inner_dict.values()]
                    row_data[f"avg_speed_val_{condition}"] = np.mean(all_speed_values) if all_speed_values else np.nan
                else:
                    row_data[f"avg_speed_val_{condition}"] = np.nan

                # Handle avg_time_val
                if time_vals:
                    all_time_values = [val for inner_dict in time_vals.values() for val in inner_dict.values()]
                    row_data[f"avg_time_val_{condition}"] = np.mean(all_time_values) if all_time_values else np.nan
                else:
                    row_data[f"avg_time_val_{condition}"] = np.nan

            # Append the row data for the current city
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)

        for continents in unique_continents:
            filtered_df = df[(df['continent_0'] == continents) | (df['continent_1'] == continents)]
            # Create a new DataFrame to average the columns across conditions
            agg_df = pd.DataFrame()

            # Loop through the columns in the original DataFrame
            for col in filtered_df.columns:
                # Extract the feature name (without condition part)
                feature_name = "_".join(col.split("_")[:-1])
                condition = col.split("_")[-1]

                # Skip columns named "continent_0" or "continent_1"
                if "continent" in feature_name:
                    continue

                # Create a new column by averaging values across conditions for the same feature
                if feature_name not in agg_df.columns:
                    # Select the columns for this feature across all conditions
                    condition_cols = [c for c in filtered_df.columns if feature_name in c]
                    agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)

            # Compute the correlation matrix on the aggregated DataFrame
            corr_matrix_avg = agg_df.corr(method='spearman')

            # Rename the variables in the correlation matrix (example: renaming keys)
            rename_dict_3 = {
                'avg_speed_val': 'Crossing speed', 'avg_time_val': 'Crossing decision time',
                'ped_cross_city': 'Crossing', 'person_city': 'Detected persons',
                'bicycle_city': 'Detected bicycles', 'car_city': 'Detected cars',
                'motorcycle_city': 'Detected motorcycles', 'bus_city': 'Detected bus',
                'truck_city': 'Detected truck', 'cross_evnt_city': 'Crossing without traffic light',
                'vehicle_city': 'Detected total number of motor vehicle', 'cellphone_city': 'Detected cellphone',
                'trf_sign_city': 'Detected traffic signs', 'gmp': 'GMP',
                'traffic_mortality': 'Traffic mortality', 'literacy_rate': 'Literacy rate', 'gini': 'Gini coefficient',
                'traffic_index': 'Traffic Index'
                }

            corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_3, index=rename_dict_3)

            # Generate the heatmap using Plotly
            fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                            color_continuous_scale='RdBu',  # Color scale (you can customize this)
                            aspect="auto",  # Automatically adjust aspect ratio
                            # title=f"Correlation matrix heatmap {continents}"  # Title of the heatmap
                            )

            fig.update_layout(coloraxis_showscale=False)

            # update font family
            fig.update_layout(font=dict(family=common.get_configs('font_family')))

            # use value from config file
            fig.update_layout(font=dict(size=common.get_configs('font_size')))

            plots_class.save_plotly_figure(fig, f"correlation_matrix_heatmap_{continents}", save_final=True)

    @staticmethod
    def find_city_id(df, video_id, start_time):
        """
        Find the city identifier (row 'id') associated with a given video ID and start time.

        This function iterates through a DataFrame where each row may reference multiple videos and
        corresponding start times (stored as lists). It returns the 'id' value from the row where both
        the video ID and the exact start time match.

        Args:
            df (pd.DataFrame): DataFrame containing at least:
                               - 'videos': a string representing a list of video IDs.
                               - 'start_time': a string representing a list of lists of start times.
                               - 'id': unique identifier for each row (e.g., city or condition group).
            video_id (str): The video filename or identifier to search for.
            start_time (float or int): The specific start time to match within the corresponding list.

        Returns:
            The value of the 'id' field in the matching row, or None if no match is found.
        """
        logger.debug(f"{video_id}: looking for city, start_time={start_time}.")

        for _, row in df.iterrows():
            # Convert comma-separated string to list of video IDs
            videos = re.findall(r"[\w-]+", row["videos"])
            # Parse the stringified list of lists
            start_times = ast.literal_eval(row["start_time"])

            if video_id in videos:
                index = videos.index(video_id)

                # Check if the provided start_time exists in the relevant sublist
                if start_time in start_times[index]:
                    return row["id"]

        # No match found
        return None

    def get_duration_segment(self, var_dict, dfs, df_mapping, num, duration=None):
        """
        Extract and save video segments based on the fastest tracked objects in provided data.

        This function processes the top-N maximum speed segments (as returned by `find_min_max_video`),
        locates the corresponding video files, computes the actual video segment durations,
        and saves both the trimmed segment and an annotated version with bounding boxes.

        Args:
            var_dict (dict): Nested dictionary of speed values by city/state/condition, video ID, and unique object ID.
            dfs (dict): Dictionary mapping video_start_time keys to DataFrames with tracking data.
            df_mapping (pd.DataFrame): A mapping DataFrame that includes metadata like FPS per video ID.
            num (int): Number of top-speed entries to process.
            duration (float, optional): If provided, use this fixed duration instead of computing from frame count.

        Returns:
            None. Video clips are saved to disk in 'saved_snaps/original' and 'saved_snaps/tracked'.
        """
        data = self.find_min_max_video(var_dict, num=num)
        if common.get_configs('min_max_videos') is False:
            return data

        # Process only the 'max' speed segments
        for city_data in data['max'].values():
            for video_start_time, inner_value in city_data.items():
                # Extract base video name and its offset
                video_name, start_offset = video_start_time.rsplit('_', 1)
                start_offset = int(start_offset)

                for unique_id, speed in inner_value.items():
                    try:
                        # Find the existing folder containing the video file
                        existing_folder = next((
                            path for path in video_paths if os.path.exists(
                                os.path.join(path, f"{video_name}.mp4"))), None)

                        if not existing_folder:
                            raise FileNotFoundError(f"Video file '{video_name}.mp4' not found in any of the specified paths.")  # noqa:E501

                        base_video_path = os.path.join(existing_folder, f"{video_name}.mp4")

                        # Load tracking DataFrame for the current video segment
                        df = dfs[video_start_time]
                        filtered_df = df[df['Unique Id'] == unique_id]

                        if filtered_df.empty:
                            return None, None  # No data found for this unique_id

                        # Determine frame-based start and end times
                        first_frame = filtered_df['Frame Count'].min()
                        last_frame = filtered_df['Frame Count'].max()

                        # Look up the frame rate (fps) using the video_start_time
                        result = values_class.find_values_with_video_id(df_mapping, video_start_time)

                        # Check if the result is None (i.e., no matching data was found)
                        if result is not None:
                            # Unpack the result since it's not None
                            fps = result[15]

                            first_time = first_frame / fps
                            last_time = last_frame / fps

                            # Adjusted start and end times
                            real_start_time = first_time + start_offset
                            if duration is None:
                                real_end_time = start_offset + last_time
                            else:
                                real_end_time = real_start_time + duration

                            # Trim and save the raw segment
                            helper.trim_video(
                                input_path=base_video_path,
                                output_path=os.path.join("saved_snaps",
                                                         "original", f"{video_name}_{real_start_time}.mp4"),
                                start_time=real_start_time,
                                end_time=real_end_time
                            )

                            # Overlay YOLO boxes on the saved segment
                            self.draw_yolo_boxes_on_video(df=filtered_df,
                                                          fps=fps,
                                                          video_path=os.path.join("saved_snaps",
                                                                                  "original",
                                                                                  f"{video_name}_{real_start_time}.mp4"),  # noqa:E501
                                                          output_path=os.path.join("saved_snaps",
                                                                                   "tracked",
                                                                                   f"{video_name}_{real_start_time}.mp4"))  # noqa:E501

                    except FileNotFoundError as e:
                        logger.error(f"Error: {e}")

        return data

    def draw_yolo_boxes_on_video(self, df, fps, video_path, output_path):
        """
        Draw YOLO-style bounding boxes on a video and save the annotated output.

        This method takes a DataFrame containing normalized bounding box coordinates (in YOLO format),
        matches them frame-by-frame to the input video, draws the corresponding boxes and labels,
        and writes the resulting video to disk.

        Args:
            df (pd.DataFrame): DataFrame containing at least the following columns:
                - 'Frame Count': Original frame indices in the source video.
                - 'X-center', 'Y-center': Normalized center coordinates (0 to 1).
                - 'Width', 'Height': Normalized width and height (0 to 1).
                - 'Unique Id': Identifier to display in the label.
            fps (float): Frames per second for the output video.
            video_path (str): Path to the input video file.
            output_path (str): Path to save the annotated output video.

        Raises:
            IOError: If the input video cannot be opened.
        """

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Normalise frame indices to start from 0
        min_frame = df["Frame Count"].min()
        df["Frame Index"] = df["Frame Count"] - min_frame

        # Attempt to open the input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Get video dimensions and total number of frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set up video writer with the same resolution and specified fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Writing to {output_path} ({width}x{height} @ {fps}fps)")

        frame_index = 0

        # Process each frame
        while frame_index < total_frames:
            success, frame = cap.read()
            if not success:
                logger.error(f"Failed to read frame {frame_index}")
                break

            # Filter YOLO data for this adjusted frame index
            frame_data = df[df["Frame Index"] == frame_index]

            for _, row in frame_data.iterrows():
                # Convert normalized coordinates to absolute pixel values
                x_center = row["X-center"] * width
                y_center = row["Y-center"] * height
                w = row["Width"] * width
                h = row["Height"] * height

                # Calculate top-left and bottom-right corners of the box
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Draw rectangle and label with unique ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID: {int(row['Unique Id'])}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the modified frame to the output video
            out.write(frame)
            frame_index += 1

        # Release video objects
        cap.release()
        out.release()

    def get_duration(self, df, video_id, start_time):
        """
        Retrieve the duration of a specific video segment from a DataFrame.

        Each row in the DataFrame contains information about multiple video segments stored as strings.
        The function looks for a specific `video_id` and `start_time`, and calculates the duration
        by finding the corresponding end time.

        Args:
            df (pd.DataFrame): A DataFrame where each row contains:
                               - 'videos': a string representation of a list of video IDs.
                               - 'start_time': a stringified list of lists of start times.
                               - 'end_time': a stringified list of lists of end times.
            video_id (str): The ID of the video segment to search for.
            start_time (float or int): The specific start time of the video segment.

        Returns:
            float: Duration of the segment (end_time - start_time), or
            None: If no matching segment is found.
        """
        for _, row in df.iterrows():
            # Extract list of video IDs
            videos = re.findall(r"[\w-]+", row["videos"])

            # Convert stringified lists of lists into Python objects
            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])

            if video_id in videos:
                index = videos.index(video_id)  # get the index of the video

                # Ensure start_time is in the corresponding start_times list
                if start_time in start_times[index]:  # check if start_time matches
                    # find end time that matches the start time
                    index_start = start_times[index].index(start_time)
                    end_time = end_times[index][index_start]
                    # Return the difference as the duration
                    return end_time - start_time

        # Return None if no matching video_id and start_time were found
        return None

    def find_min_max_video(self, var_dict, num=2):
        """
        Find the top and bottom N videos based on speed values from a nested dictionary structure.

        The function flattens a deeply nested dictionary containing speed values associated with
        unique identifiers for videos in various city/state/condition groupings. It then extracts
        the `num` largest and smallest speed values and reconstructs a dictionary preserving the original structure.

        Args:
            var_dict (dict): A nested dictionary in the format:
                            {
                                'City_State_Condition': {
                                    'video_id': {
                                        'unique_id': speed_value
                                    }
                                }
                            }
            num (int): Number of top and bottom entries to return based on speed. Default is 2.

        Returns:
            dict: A dictionary with two keys:
                - 'max': contains the top `num` speed entries
                - 'min': contains the bottom `num` speed entries
                Each follows the original nested structure.
        """
        all_speeds = []

        # Flatten the nested dictionary and collect tuples of (speed, city_state_cond, video_id, unique_id)
        for city_lat_long_cond, videos in var_dict.items():
            for video_id, unique_dict in videos.items():
                for unique_id, speed in unique_dict.items():
                    all_speeds.append((speed, city_lat_long_cond, video_id, unique_id))

        # Use heapq to efficiently get the top and bottom N entries based on speed
        top_n = heapq.nlargest(num, all_speeds, key=lambda x: x[0])
        bottom_n = heapq.nsmallest(num, all_speeds, key=lambda x: x[0])

        # Helper function to rebuild the nested structure from a list of tuples
        def format_result(entries):
            temp_result = defaultdict(lambda: defaultdict(dict))
            for speed, city_lat_long_cond, video_id, unique_id in entries:
                temp_result[city_lat_long_cond][video_id][unique_id] = speed

            # Convert defaultdicts to regular dicts for clean output
            return {
                city: {video: dict(uniq) for video, uniq in videos.items()}
                for city, videos in temp_result.items()
            }

        # Return both top and bottom results in the original structure
        return {
            'max': format_result(top_n),
            'min': format_result(bottom_n)
        }

    @staticmethod
    def scatter(df, x, y, color=None, symbol=None, size=None, text=None, trendline=None, hover_data=None,
                marker_size=None, pretty_text=False, marginal_x='violin', marginal_y='violin', xaxis_title=None,
                yaxis_title=None, xaxis_range=None, yaxis_range=None, name_file=None, save_file=False,
                save_final=False, fig_save_width=1320, fig_save_height=680, font_family=None, font_size=None,
                hover_name=None, legend_title=None, legend_x=None, legend_y=None, label_distance_factor=1.0):
        """
        Output scatter plot of variables x and y with optional assignment of colour and size.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (str): dataframe column to plot on x axis.
            y (str): dataframe column to plot on y axis.
            color (str, optional): dataframe column to assign colour of points.
            symbol (str, optional): dataframe column to assign symbol of points.
            size (str, optional): dataframe column to assign doze of points.
            text (str, optional): dataframe column to assign text labels.
            trendline (str, optional): trendline. Can be 'ols', 'lowess'
            hover_data (list, optional): dataframe columns to show on hover.
            marker_size (int, optional): size of marker. Should not be used together with size argument.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            marginal_x (str, optional): type of marginal on x axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
            hover_name (list, optional): title on top of hover popup.
            legend_title (list, optional): title on top of legend.
            legend_x (float, optional): x position of legend.
            legend_y (float, optional): y position of legend.
            label_distance_factor (float, optional): multiplier for the threshold to control density of text labels.
        """
        logger.info('Creating scatter plot for x={} and y={}.', x, y)
        # using size and marker_size is not supported
        if marker_size and size:
            logger.error('Arguments marker_size and size cannot be used together.')
            return -1
        # using marker_size with histogram marginal(s) is not supported
        if (marker_size and (marginal_x == 'histogram' or marginal_y == 'histogram')):
            logger.error('Argument marker_size cannot be used together with histogram marginal(s).')
            return -1
        # prettify text
        if pretty_text:
            if isinstance(df.iloc[0][x], str):  # check if string
                # replace underscores with spaces
                df[x] = df[x].str.replace('_', ' ')
                # capitalise
                df[x] = df[x].str.capitalize()
            if isinstance(df.iloc[0][y], str):  # check if string
                # replace underscores with spaces
                df[y] = df[y].str.replace('_', ' ')
                # capitalise
                df[y] = df[y].str.capitalize()
            if color and isinstance(df.iloc[0][color], str):  # check if string
                # replace underscores with spaces
                df[color] = df[color].str.replace('_', ' ')
                # capitalise
                df[color] = df[color].str.capitalize()
            if size and isinstance(df.iloc[0][size], str):  # check if string
                # replace underscores with spaces
                df[size] = df[size].str.replace('_', ' ')
                # capitalise
                df[size] = df[size].str.capitalize()
            try:
                # check if string
                if text and isinstance(df.iloc[0][text], str):
                    # replace underscores with spaces
                    df[text] = df[text].str.replace('_', ' ')
                    # capitalise
                    df[text] = df[text].str.capitalize()
            except ValueError as e:
                logger.debug('Tried to prettify {} with exception {}.', text, e)

        # check and clean the data
        df = df.replace([np.inf, -np.inf], np.nan).dropna()  # Remove NaNs and Infs

        if text:
            if text in df.columns:
                # use KDTree to check point density
                tree = KDTree(df[[x, y]].values)  # Ensure finite values
                distances, _ = tree.query(df[[x, y]].values, k=2)  # Find nearest neighbor distance

                # define a distance threshold for labeling
                threshold = np.mean(distances[:, 1]) * label_distance_factor

                # only label points that are not too close to others
                df["display_label"] = np.where(distances[:, 1] > threshold, df[text], "")

                text = "display_label"
            else:
                logger.warning("Column 'country' not found, skipping display_label logic.")

        # scatter plot with histograms
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            fig = px.scatter(df,
                             x=x,
                             y=y,
                             color=color,
                             symbol=symbol,
                             size=size,
                             text=text,
                             trendline=trendline,
                             hover_data=hover_data,
                             hover_name=hover_name,
                             marginal_x=marginal_x,
                             marginal_y=marginal_y)

        # font size of text labels
        for trace in fig.data:
            if trace.type == "scatter" and "text" in trace:  # type: ignore
                trace.textfont = dict(size=common.get_configs('font_size'))  # type: ignore

        # location of labels
        if not marginal_x and not marginal_y:
            fig.update_traces(textposition=Analysis.improve_text_position(df[x]))

        # update layout
        fig.update_layout(template=common.get_configs('plotly_template'),
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # change marker size
        if marker_size:
            fig.update_traces(marker=dict(size=marker_size))
        # update legend title
        if legend_title is not None:
            fig.update_layout(legend_title_text=legend_title)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=common.get_configs('font_size')))
        # legend
        if legend_x and legend_y:
            fig.update_layout(legend=dict(x=legend_x, y=legend_y, bgcolor='rgba(0,0,0,0)'))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'scatter_' + x + '-' + y
            # Final adjustments and display
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            plots_class.save_plotly_figure(fig, name_file, save_final=True)
        # open it in localhost instead
        else:
            fig.show()

    @staticmethod
    def improve_text_position(x):
        """
        Generate a list of text positions for plotting annotations based on the length of the input list `x`.

        This function alternates between predefined text positions (e.g., 'top center', 'bottom center')
        for each item in `x`. It is more efficient and visually clear if the corresponding x-values are sorted
        before using this function.

        Args:
            x (list): A list of values (typically x-axis data points).

        Returns:
            list: A list of text positions corresponding to each element in `x`.
        """
        # Predefined positions to alternate between (can be extended with more options)
        positions = ['top center', 'bottom center']

        # Cycle through the positions for each element in the input list
        return [positions[i % len(positions)] for i in range(len(x))]

    @staticmethod
    def get_coordinates(city, state, country):
        """
        Retrieve geographic coordinates (latitude and longitude) for a given city, state, and country.

        The function uses the Nominatim geocoding service from the geopy library. It first constructs a location
        query string based on the provided city, optional state, and country. A unique user-agent is generated
        for each call to avoid getting blocked by the server.

        Args:
            city (str): Name of the city.
            state (str or None): Name of the state or province. Optional; can be None or 'nan'.
            country (str): Name of the country.

        Returns:
            tuple: A tuple (latitude, longitude) if geocoding is successful, otherwise (None, None).

        Exceptions:
            - Logs and handles `GeocoderTimedOut` if the geocoding request times out.
            - Logs and handles `GeocoderUnavailable` if the geocoding server is unreachable.
        """
        # Generate a unique user agent with the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        user_agent = f"my_geocoding_script_{current_time}"

        # Initialise the Nominatim geocoder with the unique user agent
        geolocator = Nominatim(user_agent=user_agent)

        try:
            # Form the query string depending on whether a valid state value is provided
            if state and str(state).lower() != 'nan':
                location_query = f"{city}, {state}, {country}"  # Combine city, state and country
            else:
                location_query = f"{city}, {country}"  # Combine city and country
            location = geolocator.geocode(location_query, timeout=2)  # type: ignore # Set a 2-second timeout

            if location:
                return location.latitude, location.longitude  # type: ignore
            else:
                logger.error(f"Failed to geocode {location_query}")
                return None, None  # Return None if city is not found

        except GeocoderTimedOut:
            # Handle timeout errors when the request takes too long
            logger.error(f"Geocoding timed out for {location_query}.")
        except GeocoderUnavailable:
            # Handle cases where the geocoding service is not available
            logger.error(f"Geocoding server could not be reached for {location_query}.")
            return None, None  # Return None if city is not found

    @staticmethod
    def hist(data_index, name, nbins=None, color=None, pretty_text=False, marginal='rug',
             xaxis_title=None, yaxis_title=None, name_file=None, save_file=False, save_final=False,
             fig_save_width=1320, fig_save_height=680, font_family=None, font_size=None,
             vlines=None, xrange=None):
        """
        Output histogram of selected data from pickle file.

        Args:
            data_index (int): index of the item in the tuple to plot.
            nbins (int, optional): number of bins in histogram.
            color (str, optional): dataframe column to assign colour of bars.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising.
            marginal (str, optional): marginal type: 'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): whether to save HTML file of the plot.
            save_final (bool, optional): whether to save final figure to /figures.
            fig_save_width (int, optional): width of saved figure.
            fig_save_height (int, optional): height of saved figure.
            font_family (str, optional): font family to use. Defaults to config.
            font_size (int, optional): font size to use. Defaults to config.
        """

        # Load data from pickle file
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)
        nested_dict = data_tuple[data_index]

        all_values = [speed for city in nested_dict.values() for video in city.values() for speed in video.values()]

        # --- Calculate mean and median ---
        mean_val = np.mean(all_values)
        median_val = np.median(all_values)

        logger.info('Creating histogram for {}.', name)

        # Restrict values to the specified x-range if provided
        if xrange is not None:
            x_min, x_max = xrange
            all_values = [x for x in all_values if x_min <= x <= x_max]

        # Create histogram
        if color:
            fig = px.histogram(df, x=all_values, nbins=nbins, marginal=marginal, color=color)
        else:
            fig = px.histogram(df, x=all_values, nbins=nbins, marginal=marginal)

        fig.update_layout(
            xaxis=dict(tickformat='digits'),
            template=common.get_configs('plotly_template'),
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            font=dict(
                family=font_family if font_family else common.get_configs('font_family'),
                size=font_size if font_size else common.get_configs('font_size')
            )
        )

        # --- Add vertical lines for mean and median ---
        fig.add_vline(
            x=mean_val,
            line_dash='dash',
            line_color='blue',
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position='top right'
        )

        fig.add_vline(
            x=median_val,
            line_dash='dash',
            line_color='red',
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position='top left'
        )

        if vlines:
            for x in vlines:
                fig.add_vline(
                    x=x,
                    line_dash='dot',
                    line_color='black',
                    annotation_text=f'{x}',
                    annotation_position='top'
                )

        if save_file:
            if not name_file:
                name_file = f"hist_{name}"
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            plots_class.save_plotly_figure(fig, name_file, save_final=True)
        else:
            fig.show()


analysis_class = Analysis()

# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")

    if os.path.exists(file_results) and not common.get_configs('always_analyse'):
        # Load the data from the pickle file
        with open(file_results, 'rb') as file:
            (data, person_counter, bicycle_counter, car_counter, motorcycle_counter,
             bus_counter, truck_counter, cellphone_counter, traffic_light_counter, stop_sign_counter,
             pedestrian_cross_city, pedestrian_crossing_count, person_city, bicycle_city, car_city,
             motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city,
             cellphone_city, traffic_sign_city, speed_values, time_values, avg_time, avg_speed,
             df_mapping, with_trf_light, without_trf_light, min_max_speed, min_max_time) = pickle.load(file)

        logger.info("Loaded analysis results from pickle file.")
    else:
        # Store the mapping file
        df_mapping = pd.read_csv(common.get_configs("mapping"))

        # Produce map with all data
        df = df_mapping.copy()  # copy df to manipulate for output
        df['state'] = df['state'].fillna('NA')  # Set state to NA

        # Sort by continent and city, both in ascending order
        df = df.sort_values(by=["continent", "city"], ascending=[True, True])

        # Data to avoid showing on hover in scatter plots
        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type',
                          'channel']
        hover_data = list(set(df.columns) - set(columns_remove))
        Analysis.get_mapbox_map(df=df, hover_data=hover_data, file_name='mapbox_map_all')  # mapbox map with all data

        # Get the population threshold from the configuration
        population_threshold = common.get_configs("population_threshold")

        # Get the minimum percentage of country population from the configuration
        min_percentage = common.get_configs("min_city_population_percentage")

        # Filter df_mapping to include cities that meet either of the following criteria:
        # 1. The city's population is greater than the threshold
        # 2. The city's population is at least the minimum percentage of the country's population
        df_mapping = df_mapping[
            (df_mapping["population_city"] > population_threshold) |  # Condition 1
            (df_mapping["population_city"] >= min_percentage * df_mapping["population_country"])  # Condition 2
        ]

        # Limit countries if required
        countries_include = common.get_configs("countries_analyse")
        if countries_include:
            df_mapping = df_mapping[df_mapping["iso3"].isin(common.get_configs("countries_analyse"))]

        pedestrian_crossing_count, data = {}, {}
        person_counter, bicycle_counter, car_counter, motorcycle_counter = 0, 0, 0, 0
        bus_counter, truck_counter, cellphone_counter, traffic_light_counter, stop_sign_counter = 0, 0, 0, 0, 0

        total_duration = Analysis.calculate_total_seconds(df_mapping)
        logger.info(f"Duration of videos in seconds: {total_duration}, in minutes: {total_duration/60:.2f}, in " +
                    f"hours: {total_duration/60/60:.2f}.")
        logger.info("Total number of videos: {}.", Analysis.calculate_total_videos(df_mapping))
        country, number = Analysis.get_unique_values(df_mapping, "country")
        logger.info("Total number of countries: {}.", number)
        city, number = Analysis.get_unique_values(df_mapping, "city")
        logger.info("Total number of cities: {}.", number)

        # Stores the content of the csv file in form of {name_time: content}
        dfs = Analysis.read_csv_files(common.get_configs('data'), df_mapping)

        # add information for each city to then be appended to mapping
        df_mapping['person'] = 0
        df_mapping['bicycle'] = 0
        df_mapping['car'] = 0
        df_mapping['motorcycle'] = 0
        df_mapping['bus'] = 0
        df_mapping['truck'] = 0
        df_mapping['cellphone'] = 0
        df_mapping['traffic_light'] = 0
        df_mapping['stop_sign'] = 0
        df_mapping['total_time'] = 0
        df_mapping['speed_crossing'] = 0.0
        df_mapping['speed_crossing_day'] = 0.0
        df_mapping['speed_crossing_night'] = 0.0
        df_mapping['time_crossing'] = 0.0
        df_mapping['time_crossing_day'] = 0.0
        df_mapping['time_crossing_night'] = 0.0
        df_mapping['with_trf_light_day'] = 0.0
        df_mapping['with_trf_light_night'] = 0.0
        df_mapping['without_trf_light_day'] = 0.0
        df_mapping['without_trf_light_night'] = 0.0

        # Loop over rows of data
        logger.info("Analysing data.")
        for key, value in tqdm(dfs.items(), total=len(dfs)):
            # extract information for the csv file from mapping
            logger.debug(f"{key}: fetching values.")
            video_id, start_index = key.rsplit("_", 1)  # split to extract id and index
            video_city_id = Analysis.find_city_id(df_mapping, video_id, int(start_index))
            video_city = df_mapping.loc[df_mapping["id"] == video_city_id, "city"].values[0]  # type:ignore
            video_state = df_mapping.loc[df_mapping["id"] == video_city_id, "state"].values[0]  # type:ignore
            video_country = df_mapping.loc[df_mapping["id"] == video_city_id, "country"].values[0]  # type:ignore
            logger.debug(f"{key}: found values {video_city}, {video_state}, {video_country}.")

            # Get the number of number and unique id of the object crossing the road
            ids = Analysis.pedestrian_crossing(dfs[key],
                                               common.get_configs("boundary_left"),
                                               common.get_configs("boundary_right"),
                                               0)

            # Saving it in a dictionary in: {video-id_time: count, ids}
            pedestrian_crossing_count[key] = {"ids": ids}

            # Saves the time to cross in form {name_time: {id(s): time(s)}}
            data[key] = algorithms_class.time_to_cross(dfs[key], pedestrian_crossing_count[key]["ids"], key)

            # Calculate the total number of different objects detected
            person_video = Analysis.count_object(dfs[key], 0)
            person_counter += person_video
            df_mapping.loc[df_mapping["id"] == video_city_id, "person"] += person_video

            bicycle_video = Analysis.count_object(dfs[key], 1)
            bicycle_counter += bicycle_video
            df_mapping.loc[df_mapping["id"] == video_city_id, "bicycle"] += bicycle_video

            car_video = Analysis.count_object(dfs[key], 2)
            car_counter += car_video
            df_mapping.loc[df_mapping["id"] == video_city_id, "car"] += car_video

            motorcycle_video = Analysis.count_object(dfs[key], 3)
            motorcycle_counter += motorcycle_video
            df_mapping.loc[df_mapping["id"] == video_city_id, "motorcycle"] += motorcycle_video

            bus_video = Analysis.count_object(dfs[key], 5)
            bus_counter += bus_video
            df_mapping.loc[df_mapping["id"] == video_city_id, "bus"] += bus_video

            truck_video = Analysis.count_object(dfs[key], 7)
            truck_counter += truck_video
            df_mapping.loc[df_mapping["id"] == video_city_id, "truck"] += truck_video

            cellphone_video = Analysis.count_object(dfs[key], 67)
            cellphone_counter += cellphone_video
            df_mapping.loc[df_mapping["id"] == video_city_id, "cellphone"] += cellphone_video

            traffic_light_video = Analysis.count_object(dfs[key], 9)
            traffic_light_counter += traffic_light_video
            df_mapping.loc[df_mapping["id"] == video_city_id, "traffic_light"] += traffic_light_video

            stop_sign_video = Analysis.count_object(dfs[key], 11)
            stop_sign_counter += stop_sign_video
            df_mapping.loc[df_mapping["id"] == video_city_id, "stop_sign"] += stop_sign_video

            # add duration of segment
            time_video = analysis_class.get_duration(df_mapping, video_id, int(start_index))
            df_mapping.loc[df_mapping["id"] == video_city_id, "total_time"] += time_video  # type: ignore

        # Aggregated values
        logger.info("Calculating aggregated values for crossing speed.")
        speed_values = algorithms_class.calculate_speed_of_crossing(df_mapping, dfs, data)
        avg_speed = algorithms_class.avg_speed_of_crossing(df_mapping, dfs, data)

        # Add to mapping file
        for key, value in tqdm(avg_speed.items(), total=len(avg_speed)):
            parts = key.split("_")
            city = parts[0]  # First part is city
            lat = parts[1]  # Second part is latitude
            long = parts[2]  # Third part is longitude
            time_of_day = int(parts[3])  # Fourth part is the time-of-day

            # state = parts[1] if parts[1] != "unknown" else np.nan
            state = values_class.get_value(df_mapping, "city", city, "lat", lat, "state")

            if not time_of_day:  # day
                df_mapping.loc[
                    (df_mapping["city"] == city) &
                    ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                    "speed_crossing_day"
                ] = float(value)  # Explicitly cast speed to float
            else:  # night
                df_mapping.loc[
                    (df_mapping["city"] == city) &
                    ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                    "speed_crossing_night"
                ] = float(value)  # Explicitly cast speed to float

        # calculate average values
        df_mapping["speed_crossing"] = np.where(
            (df_mapping["speed_crossing_day"] > 0) & (df_mapping["speed_crossing_night"] > 0),
            df_mapping[["speed_crossing_day", "speed_crossing_night"]].mean(axis=1),
            np.where(
                df_mapping["speed_crossing_day"] > 0, df_mapping["speed_crossing_day"],
                np.where(df_mapping["speed_crossing_night"] > 0, df_mapping["speed_crossing_night"], np.nan)
            )
        )
        logger.info("Calculating aggregated values for crossing decision time.")
        time_values = algorithms_class.time_to_start_cross(df_mapping, dfs, data)
        avg_time = algorithms_class.avg_time_to_start_cross(df_mapping, dfs, data)

        # add to mapping file
        for key, value in tqdm(avg_time.items(), total=len(avg_time)):
            parts = key.split("_")
            city = parts[0]  # First part is city
            lat = parts[1]  # Second part is latitude
            long = parts[2]  # Third part is longitude
            time_of_day = int(parts[3])  # Fourth part is the time-of-day

            if not time_of_day:  # day
                df_mapping.loc[
                    (df_mapping["city"] == city) &
                    ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                    "time_crossing_day"
                ] = float(value)  # Explicitly cast speed to float
            else:  # night
                df_mapping.loc[
                    (df_mapping["city"] == city) &
                    ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                    "time_crossing_night"
                ] = float(value)  # Explicitly cast speed to float

        # calculate average values
        df_mapping["time_crossing"] = np.where(
            (df_mapping["time_crossing_day"] > 0) & (df_mapping["time_crossing_night"] > 0),
            df_mapping[["time_crossing_day", "time_crossing_night"]].mean(axis=1),
            np.where(
                df_mapping["time_crossing_day"] > 0, df_mapping["time_crossing_day"],
                np.where(df_mapping["time_crossing_night"] > 0, df_mapping["time_crossing_night"], np.nan)
            )
        )

        min_max_speed = analysis_class.get_duration_segment(speed_values, dfs, df_mapping, num=10, duration=None)
        min_max_time = analysis_class.get_duration_segment(time_values, dfs, df_mapping, num=10, duration=None)

        # TODO: these functions are slow, and they are possibly not needed now as counts are added to df_mapping
        logger.info("Calculating counts of detected traffic signs.")
        traffic_sign_city = analysis_class.calculate_traffic_signs(df_mapping, dfs)
        logger.info("Calculating counts of detected mobile phones.")
        cellphone_city = Analysis.calculate_cell_phones(df_mapping, dfs)
        logger.info("Calculating counts of detected vehicles.")
        vehicle_city = analysis_class.calculate_traffic(df_mapping, dfs, motorcycle=1, car=1, bus=1, truck=1)
        logger.info("Calculating counts of detected bicycles.")
        bicycle_city = analysis_class.calculate_traffic(df_mapping, dfs, bicycle=1)
        logger.info("Calculating counts of detected cars (subset of vehicles).")
        car_city = analysis_class.calculate_traffic(df_mapping, dfs, car=1)
        logger.info("Calculating counts of detected motorcycles (subset of vehicles).")
        motorcycle_city = analysis_class.calculate_traffic(df_mapping, dfs, motorcycle=1)
        logger.info("Calculating counts of detected buses (subset of vehicles).")
        bus_city = analysis_class.calculate_traffic(df_mapping, dfs, bus=1)
        logger.info("Calculating counts of detected trucks (subset of vehicles).")
        truck_city = analysis_class.calculate_traffic(df_mapping, dfs, truck=1)
        logger.info("Calculating counts of detected persons.")
        person_city = analysis_class.calculate_traffic(df_mapping, dfs, person=1)
        logger.info("Calculating counts of detected crossing events with traffic lights.")
        cross_evnt_city = Analysis.crossing_event_wt_traffic_light(df_mapping, dfs, data)
        logger.info("Calculating counts of crossing events.")
        pedestrian_cross_city = Analysis.pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping)

        # Jaywalking data
        logger.info("Calculating parameters for detection of jaywalking.")
        with_trf_light, without_trf_light, _ = Analysis.crossing_event_wt_traffic_equipment(df_mapping, dfs, data)
        for key, value in with_trf_light.items():
            parts = key.split("_")
            city = parts[0]  # First part is city
            lat = parts[1]  # Second part is latitude
            long = parts[2]  # Third part is longitude
            time_of_day = int(parts[3])  # Fourth part is the time-of-day

            if not time_of_day:  # day
                df_mapping.loc[
                    (df_mapping["city"] == city) &
                    ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                    "with_trf_light_day"
                ] = int(value)  # Explicitly cast to int
            else:  # night
                df_mapping.loc[
                    (df_mapping["city"] == city) &
                    ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                    "with_trf_light_night"
                ] = int(value)  # Explicitly cast to int

        # add to mapping file
        for key, value in without_trf_light.items():
            parts = key.split("_")
            city = parts[0]  # First part is city
            lat = parts[1]  # Second part is latitude
            long = parts[2]  # Third part is longitude
            time_of_day = int(parts[3])  # Fourth part is the time-of-day

            if not time_of_day:  # day
                df_mapping.loc[
                    (df_mapping["city"] == city) &
                    ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                    "without_trf_light_day"
                ] = int(value)  # Explicitly cast to int
            else:  # night
                df_mapping.loc[
                    (df_mapping["city"] == city) &
                    ((df_mapping["state"] == state) | (pd.isna(df_mapping["state"]) & pd.isna(state))),
                    "without_trf_light_night"
                ] = int(value)  # Explicitly cast to int

        # Add column with count of videos
        df_mapping["total_videos"] = df_mapping["videos"].apply(lambda x: len(x.strip("[]").split(",")) if x.strip("[]") else 0)  # noqa: E501

        # Get lat and lon for cities
        logger.info("Fetching lat and lon coordinates for cities.")
        for index, row in tqdm(df_mapping.iterrows(), total=len(df_mapping)):
            if pd.isna(row["lat"]) or pd.isna(row["lon"]):
                lat, lon = Analysis.get_coordinates(row["city"],
                                                    row["state"],
                                                    common.correct_country(row["country"]))  # type: ignore
                df_mapping.at[index, 'lat'] = lat
                df_mapping.at[index, 'lon'] = lon

        # Save the results to a pickle file
        logger.info("Saving results to a pickle file {}.", file_results)
        with open(file_results, 'wb') as file:
            pickle.dump((data,                       # 0
                         person_counter,             # 1
                         bicycle_counter,            # 2
                         car_counter,                # 3
                         motorcycle_counter,         # 4
                         bus_counter,                # 5
                         truck_counter,              # 6
                         cellphone_counter,          # 7
                         traffic_light_counter,      # 8
                         stop_sign_counter,          # 9
                         pedestrian_cross_city,      # 10
                         pedestrian_crossing_count,  # 11
                         person_city,                # 12
                         bicycle_city,               # 13
                         car_city,                   # 14
                         motorcycle_city,            # 15
                         bus_city,                   # 16
                         truck_city,                 # 17
                         cross_evnt_city,            # 18
                         vehicle_city,               # 19
                         cellphone_city,             # 20
                         traffic_sign_city,          # 21
                         speed_values,               # 22
                         time_values,                # 23
                         avg_time,                   # 24
                         avg_speed,                  # 25
                         df_mapping,                 # 26
                         with_trf_light,             # 27
                         without_trf_light,          # 28
                         min_max_speed,              # 29
                         min_max_time),              # 30
                        file)
        logger.info("Analysis results saved to pickle file.")

    # Set index as ID
    df_mapping = df_mapping.set_index("id")
    # Sort by continent and city, both in ascending order
    df_mapping = df_mapping.sort_values(by=["continent", "city"], ascending=[True, True])
    # Save updated mapping file in output
    os.makedirs(common.output_dir, exist_ok=True)  # check if folder
    df_mapping.to_csv(os.path.join(common.output_dir, "mapping_updated.csv"))

    logger.info("Detected:")
    logger.info(f"person: {person_counter}; bicycle: {bicycle_counter}; car: {car_counter}")
    logger.info(f"motorcycle: {motorcycle_counter}; bus: {bus_counter}; truck: {truck_counter}")
    logger.info(f"cellphone: {cellphone_counter}; traffic light: {traffic_light_counter}; " +
                f"traffic sign: {stop_sign_counter}")

    logger.info("Producing output.")
    # Data to avoid showing on hover in scatter plots
    columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'fps_list', 'vehicle_type',
                      'channel']
    hover_data = list(set(df_mapping.columns) - set(columns_remove))

    # Analysis.get_world_map(df_mapping)
    df = df_mapping.copy()  # copy df to manipulate for output
    df['state'] = df['state'].fillna('NA')  # Set state to NA

    Analysis.get_mapbox_map(df=df, hover_data=hover_data)  # mapbox map
    Analysis.get_world_map(df_mapping=df)  # map with countries

    # Amount of footage
    Analysis.scatter(df=df,
                     x="total_time",
                     y="person",
                     color="continent",
                     text="city",
                     xaxis_title='Total time of footage (s)',
                     yaxis_title='Number of detected pedestrians',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.01,
                     legend_y=1.0,
                     label_distance_factor=5.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    Analysis.hist(data_index=22, name="speed", save_file=True)

    # todo: ISO-3 codes next to figures shift. need to correct once "final" dataset is online
    Analysis.speed_and_time_to_start_cross(df_mapping,
                                           x_axis_title_height=110,
                                           font_size_captions=common.get_configs("font_size") + 8,
                                           legend_x=0.9,
                                           legend_y=0.01,
                                           legend_spacing=0.0026)

    plots_class.stack_plot(df,
                           order_by="alphabetical",
                           metric="time",
                           data_view="combined",
                           title_text="Time to start crossing (s)",
                           filename="time_crossing_alphabetical",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

    plots_class.stack_plot(df,
                           order_by="alphabetical",
                           metric="time",
                           data_view="day",
                           title_text="Time to start crossing (s)",
                           filename="time_crossing_alphabetical_day",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

    plots_class.stack_plot(df,
                           order_by="alphabetical",
                           metric="time",
                           data_view="night",
                           title_text="Time to start crossing (s)",
                           filename="time_crossing_alphabetical_night",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

    plots_class.stack_plot(df,
                           order_by="average",
                           metric="time",
                           data_view="combined",
                           title_text="Time to start crossing (s)",
                           filename="time_crossing_avg",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

    plots_class.stack_plot(df,
                           order_by="average",
                           metric="time",
                           data_view="day",
                           title_text="Time to start crossing (s)",
                           filename="time_crossing_avg_day",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

    plots_class.stack_plot(df,
                           order_by="average",
                           metric="time",
                           data_view="night",
                           title_text="Time to start crossing (s)",
                           filename="time_crossing_avg_night",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

    plots_class.stack_plot(df,
                           order_by="alphabetical",
                           metric="speed",
                           data_view="combined",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_alphabetical",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

    plots_class.stack_plot(df,
                           order_by="alphabetical",
                           metric="speed",
                           data_view="day",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_alphabetical_day",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

    plots_class.stack_plot(df,
                           order_by="alphabetical",
                           metric="speed",
                           data_view="night",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_alphabetical_night",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=80,
                           right_margin=80
                           )

    plots_class.stack_plot(df,
                           order_by="average",
                           metric="speed",
                           data_view="combined",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_avg",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

    plots_class.stack_plot(df,
                           order_by="average",
                           metric="speed",
                           data_view="day",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_avg_day",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

    plots_class.stack_plot(df,
                           order_by="average",
                           metric="speed",
                           data_view="night",
                           title_text="Mean speed of crossing (in m/s)",
                           filename="speed_crossing_avg_night",
                           font_size_captions=common.get_configs("font_size") + 8,
                           left_margin=10,
                           right_margin=10
                           )

    Analysis.correlation_matrix(df_mapping)

    # Speed of crossing vs time to start crossing
    df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
    df = df[df["time_crossing"] != 0]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="speed_crossing",
                     y="time_crossing",
                     color="continent",
                     text="city",
                     xaxis_title='Mean speed of crossing (in m/s)',
                     yaxis_title='Mean time to start crossing (in s)',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.01,
                     legend_y=1.0,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Speed of crossing during daytime vs time to start crossing during daytime
    df = df_mapping[df_mapping["speed_crossing_day"] != 0].copy()
    df = df[df["time_crossing_day"] != 0]
    df['state'] = df['state'].fillna('NA')

    Analysis.scatter(df=df,
                     x="speed_crossing_day",
                     y="time_crossing_day",
                     color="continent",
                     text="city",
                     xaxis_title='Crossing speed during daytime (in m/s)',
                     yaxis_title='Crossing decision time during daytime (in s)',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.01,
                     legend_y=1.0,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Speed of crossing during night time vs time to start crossing during night time
    df = df_mapping[df_mapping["speed_crossing_night"] != 0].copy()
    df = df[df["time_crossing_night"] != 0]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="speed_crossing_night",
                     y="time_crossing_night",
                     color="continent",
                     text="city",
                     xaxis_title='Crossing speed during night time (in m/s)',
                     yaxis_title='Crossing decision time during night time (in s)',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=0.8,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Time to start crossing vs population of city
    df = df_mapping[df_mapping["time_crossing"] != 0].copy()
    df = df[(df["population_city"].notna()) & (df["population_city"] != 0)]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="time_crossing",
                     y="population_city",
                     color="continent",
                     text="city",
                     xaxis_title='Mean time to start crossing (in s)',
                     yaxis_title='Population of city',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=2.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Speed of crossing vs population of city
    df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
    df = df[(df["population_city"].notna()) & (df["population_city"] != 0)]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="speed_crossing",
                     y="population_city",
                     color="continent",
                     text="city",
                     xaxis_title='Mean speed of crossing (in m/s)',
                     yaxis_title='Population of city',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Time to start crossing vs population of city
    df = df_mapping[df_mapping["time_crossing"] != 0].copy()
    df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="time_crossing",
                     y="traffic_mortality",
                     color="continent",
                     text="city",
                     xaxis_title='Mean time to start crossing (in s)',
                     yaxis_title='National traffic mortality rate (per 100,000 of population)',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Speed of crossing vs population of city
    df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
    df = df[(df["traffic_mortality"].notna()) & (df["traffic_mortality"] != 0)]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="speed_crossing",
                     y="traffic_mortality",
                     color="continent",
                     text="city",
                     xaxis_title='Mean speed of crossing (in m/s)',
                     yaxis_title='National traffic mortality rate (per 100,000 of population)',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=2.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Time to start crossing vs population of city
    df = df_mapping[df_mapping["time_crossing"] != 0].copy()
    df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="time_crossing",
                     y="literacy_rate",
                     color="continent",
                     text="city",
                     xaxis_title='Mean time to start crossing (in s)',
                     yaxis_title='Literacy rate',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=0.01,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Speed of crossing vs population of city
    df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
    df = df[(df["literacy_rate"].notna()) & (df["literacy_rate"] != 0)]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="speed_crossing",
                     y="literacy_rate",
                     color="continent",
                     text="city",
                     xaxis_title='Mean speed of crossing (in m/s)',
                     yaxis_title='Literacy rate',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=0.01,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Time to start crossing vs population of city
    df = df_mapping[df_mapping["time_crossing"] != 0].copy()
    df = df[(df["gini"].notna()) & (df["gini"] != 0)]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="time_crossing",
                     y="gini",
                     color="continent",
                     text="city",
                     xaxis_title='Mean time to start crossing (in s)',
                     yaxis_title='Gini coefficient',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Speed of crossing vs population of city
    df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
    df = df[(df["gini"].notna()) & (df["gini"] != 0)]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="speed_crossing",
                     y="gini",
                     color="continent",
                     text="city",
                     xaxis_title='Mean speed of crossing (in m/s)',
                     yaxis_title='Gini coefficient',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=2.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Time to start crossing vs population of city
    df = df_mapping[df_mapping["time_crossing"] != 0].copy()
    df = df[(df["traffic_index"].notna()) & (df["traffic_index"] != 0)]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="time_crossing",
                     y="traffic_index",
                     color="continent",
                     text="city",
                     # size="gmp",
                     xaxis_title='Mean time to start crossing (in s)',
                     yaxis_title='Traffic index',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=2.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Speed of crossing vs population of city
    df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
    df = df[df["traffic_index"] != 0]
    df['state'] = df['state'].fillna('NA')
    Analysis.scatter(df=df,
                     x="speed_crossing",
                     y="traffic_index",
                     color="continent",
                     text="city",
                     xaxis_title='Mean speed of crossing (in m/s)',
                     yaxis_title='Traffic index',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=2.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Speed of crossing vs detected mobile phones
    df = df_mapping[df_mapping["time_crossing"] != 0].copy()
    df['state'] = df['state'].fillna('NA')
    df['cellphone_normalised'] = df['cellphone'] / df['total_time']
    Analysis.scatter(df=df,
                     x="time_crossing",
                     y="cellphone_normalised",
                     color="continent",
                     text="city",
                     xaxis_title='Mean time to start crossing (in s)',
                     yaxis_title='Mobile phones detected (normalised over time)',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Speed of crossing vs detected mobile phones
    df = df_mapping[df_mapping["speed_crossing"] != 0].copy()
    df['state'] = df['state'].fillna('NA')
    df['cellphone_normalised'] = df['cellphone'] / df['total_time']
    Analysis.scatter(df=df,
                     x="speed_crossing",
                     y="cellphone_normalised",
                     color="continent",
                     text="city",
                     xaxis_title='Mean speed of crossing (in m/s)',
                     yaxis_title='Mobile phones detected (normalised over time)',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore

    # Jaywalking
    Analysis.plot_crossing_without_traffic_light(df_mapping,
                                                 x_axis_title_height=60,
                                                 font_size_captions=common.get_configs("font_size"),
                                                 legend_x=0.97,
                                                 legend_y=1.0,
                                                 legend_spacing=0.004)
    Analysis.plot_crossing_with_traffic_light(df_mapping,
                                              x_axis_title_height=60,
                                              font_size_captions=common.get_configs("font_size"),
                                              legend_x=0.97,
                                              legend_y=1.0,
                                              legend_spacing=0.004)
    # Crossing with and without traffic lights
    df = df_mapping.copy()
    df['state'] = df['state'].fillna('NA')
    df['with_trf_light_norm'] = (df['with_trf_light_day'] + df['with_trf_light_night']) / df['total_time'] / df['population_city']  # noqa: E501
    df['without_trf_light_norm'] = (df['without_trf_light_day'] + df['without_trf_light_night']) / df['total_time'] / df['population_city']  # noqa: E501
    Analysis.scatter(df=df,
                     x="with_trf_light_norm",
                     y="without_trf_light_norm",
                     color="continent",
                     text="city",
                     xaxis_title='Crossing events with traffic lights (normalised)',
                     yaxis_title='Crossing events without traffic lights (normalised)',
                     pretty_text=False,
                     marker_size=10,
                     save_file=True,
                     hover_data=hover_data,
                     hover_name="city",
                     legend_title="",
                     legend_x=0.87,
                     legend_y=1.0,
                     label_distance_factor=3.0,
                     marginal_x=None,  # type: ignore
                     marginal_y=None)  # type: ignore
