# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import math
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import common
from custom_logger import CustomLogger
from logmod import logs
import statistics
import ast
from collections import defaultdict
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import pickle
from datetime import datetime
import plotly as py
import pycountry
import random
from tqdm import tqdm
import re
import warnings
from scipy.spatial import KDTree

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# set template for plotly output
template = common.get_configs('plotly_template')

# File to store the city coordinates
pickle_file_coordinates = 'city_coordinates.pkl'
pickle_file_path = 'analysis_results.pkl'

# Colours in graphs
bar_colour_1 = 'rgb(251, 180, 174)'
bar_colour_2 = 'rgb(179, 205, 227)'
bar_colour_3 = 'rgb(204, 235, 197)'
bar_colour_4 = 'rgb(222, 203, 228)'


class Analysis():

    def __init__(self) -> None:
        pass

    # Read the csv files and stores them as a dictionary in form {Unique_id : CSV}
    @staticmethod
    def read_csv_files(folder_path):
        """Reads all CSV files in a specified folder and returns their contents as a dictionary.

        Args:
            folder_path (str): Path to the folder where the CSV files are stored.

        Returns:
            dict: A dictionary where keys are CSV file names and values are DataFrames containing the
            content of each CSV file.
    """

        # Initialize an empty dictionary to store DataFrames
        dfs = {}

        # Iterate through files in the folder
        for file in os.listdir(folder_path):
            # Check if the file is a CSV file
            if file.endswith(".csv"):
                # Read the CSV file into a DataFrame
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)

                # Extract the filename without extension
                filename = os.path.splitext(file)[0]

                # Add the DataFrame to the dictionary with the filename as key
                dfs[filename] = df

        return dfs

    @staticmethod
    def count_object(dataframe, id):
        """Counts the number of unique instances of an object with a specific ID in a DataFrame.

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
    def save_plotly_figure(fig, filename, scatter_plot_flag=False, width=1600, height=900, scale=3, save_final=True):
        """Saves a Plotly figure as HTML, PNG, SVG, and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): Width of the PNG and EPS images in pixels. Defaults to 1600.
            height (int, optional): Height of the PNG and EPS images in pixels. Defaults to 900.
            scale (int, optional): Scaling factor for the PNG image. Defaults to 3.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # Create directory if it doesn't exist
        output_folder = "_output"
        output_final = "figures"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)

        # Save as HTML
        # fig.write_html(os.path.join(output_folder, filename + ".html"))
        py.offline.plot(fig, filename=os.path.join(output_folder, filename + ".html"))
        # also save the final figure
        if save_final:
            py.offline.plot(fig, filename=os.path.join(output_final, filename + ".html"),  auto_open=False)

        # remove white margins
        fig.update_layout(margin=dict(l=2, r=2, t=20, b=12))

        # Save as PNG
        if scatter_plot_flag:
            fig.write_image(os.path.join(output_folder, filename + ".png"), scale=scale)
            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(output_final, filename + ".png"), scale=scale)
        else:
            fig.write_image(os.path.join(output_folder, filename + ".png"), width=width, height=height, scale=scale)
            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(output_final, filename + ".png"), width=width, height=height, scale=scale)

        # Save as EPS
        fig.write_image(os.path.join(output_folder, filename + ".eps"), width=width, height=height)
        # also save the final figure
        if save_final:
            fig.write_image(os.path.join(output_final, filename + ".eps"), width=width, height=height)

    @staticmethod
    def adjust_annotation_positions(annotations):
        """Adjusts the positions of annotations to avoid overlap.

        Args:
            annotations (list): List of dictionaries representing annotations.

        Returns:
            list: Adjusted annotations where positions are modified to avoid overlap.
        """
        adjusted_annotations = []

        # Iterate through each annotation
        for i, ann in enumerate(annotations):
            adjusted_ann = ann.copy()

            # Adjust x and y coordinates to avoid overlap with other annotations
            for other_ann in adjusted_annotations:
                if (abs(ann['x'] - other_ann['x']) < 0.2) and (abs(ann['y'] - other_ann['y']) < 0.2):
                    adjusted_ann['y'] += 0.01  # Adjust y-coordinate (can be modified as needed)

            # Append the adjusted annotation to the list
            adjusted_annotations.append(adjusted_ann)

        return adjusted_annotations

    @staticmethod
    def plot_scatter_diag(x, y, size, color, symbol, city, plot_name, x_label, y_label,
                          legend_x=0.887, legend_y=0.986, legend_font_size=24, tick_font_size=24,
                          label_font_size=24, need_annotations=False, density_threshold=1):
        """Plots a scatter plot with diagonal markers and annotations for city locations.

        Args:
            x (list): X-axis values.
            y (dict): Dictionary containing Y-axis values with city names as keys.
            size (list): Size of markers, representing GMP per capita.
            color (list): Color of markers, representing continents.
            symbol (list): Symbol of markers, representing day/night.
            city (list): List of city names.
            plot_name (str): Name of the plot.
            x_label (str): Label for the X-axis.
            y_label (str): Label for the Y-axis.
            legend_x (float, optional): X-coordinate for the legend. Defaults to 0.887.
            legend_y (float, optional): Y-coordinate for the legend. Defaults to 0.986.
            legend_font_size (int, optional): Font size for the legend. Defaults to 12.
            tick_font_size (int, optional): Font size for the axis ticks. Defaults to 10.
            label_font_size (int, optional): Font size for axis labels. Defaults to 14.
        """
        # Hard coded colors for continents
        continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange',
                            'South America': 'purple', 'Oceania': 'brown'}

        # Create the scatter plot with hover_data for additional information (continents and sizes)
        fig = px.scatter(x=x, y=list(y.values()), size=size, color=color, symbol=symbol,
                         labels={"color": "Continent"}, color_discrete_map=continent_colors,
                         hover_data={"City": Analysis.format_city_state(city)}, size_max=10)

        # Customize the hovertemplate to only show the fields you want
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><extra></extra>"
        )

        # Hide legend for all traces generated by Plotly Express
        for trace in fig.data:
            trace.showlegend = False  # type: ignore

        # Adding labels and title with custom font sizes
        fig.update_layout(
            xaxis_title=dict(text=x_label, font=dict(size=label_font_size)),
            yaxis_title=dict(text=y_label, font=dict(size=label_font_size))
        )

        # Add markers for continents
        for continent, color_ in continent_colors.items():
            if continent in color:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                         marker=dict(color=color_), name=continent))

        # Adding manual legend for symbols
        # symbols_legend = {'diamond': 'Night', 'circle': 'Day'}
        # for symbol, description in symbols_legend.items():
        #     fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
        #                              marker=dict(symbol=symbol, color='rgba(0,0,0,0)',
        #                                          line=dict(color='black', width=2)), name=description))

        # Adding annotations for locations
        def custom_round(value, base):
            return round(value / base) * base

        annotations = []
        if need_annotations:
            point_counts = {}
            for i, key in enumerate(y.keys()):
                grid_x = custom_round(x[i], 0.1)  # Define a grid resolution for density calculation
                grid_y = custom_round(list(y.values())[i], 0.15)
                grid_point = (grid_x, grid_y)
                point_counts[grid_point] = point_counts.get(grid_point, 0) + 1

                # Show annotations for isolated points
                if point_counts[grid_point] == 1:
                    annotations.append(
                        dict(x=x[i], y=list(y.values())[i] + 0.2, text=Analysis.format_city_state(city[i]),
                             showarrow=False)
                    )
                # For dense regions, show only a fraction of the annotations
                elif point_counts[grid_point] <= density_threshold and random.random() > 0.5:
                    annotations.append(
                        dict(x=x[i], y=list(y.values())[i], text=Analysis.format_city_state(city[i]), showarrow=False)
                    )

        # Adjust annotation positions to avoid overlap
        adjusted_annotations = Analysis.adjust_annotation_positions(annotations)
        fig.update_layout(annotations=adjusted_annotations)

        # Set template
        fig.update_layout(template=template)

        # Remove legend title
        fig.update_layout(legend_title_text='')

        # Update legend position and font size
        fig.update_layout(
            legend=dict(x=legend_x, y=legend_y, traceorder="normal", font=dict(size=legend_font_size))
        )

        # Update axis tick font size
        fig.update_layout(
            xaxis=dict(tickfont=dict(size=tick_font_size)),
            yaxis=dict(tickfont=dict(size=tick_font_size))
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Save the plot
        Analysis.save_plotly_figure(fig, plot_name, scatter_plot_flag=True)

    @staticmethod
    def find_values_with_video_id(df, key):
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
                - Country
                - GDP per capita
                - Population
                - Population of the country
                - Traffic mortality
                - Continent
                - Literacy rate
                - Average height
                - ISO-3 code for country
        """

        id, start_ = key.rsplit("_", 1)  # Splitting the key into video ID and start time

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Extracting data from the DataFrame row
            video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])
            time_of_day = ast.literal_eval(row["time_of_day"])
            city = row["city"]
            state = row['state'] if not pd.isna(row['state']) else "unknown"
            country = row["country"]
            gdp = row["gdp_city_(billion_US)"]
            population = row["population_city"]
            population_country = row["population_country"]
            traffic_mortality = row["traffic_mortality"]
            continent = row["continent"]
            literacy_rate = row["literacy_rate"]
            avg_height = row["avg_height"]
            iso_country = row["ISO_country"]
            fps_list = ast.literal_eval(row["fps_list"])

            # Iterate through each video, start time, end time, and time of day
            for video, start, end, time_of_day_, fps in zip(video_ids, start_times, end_times, time_of_day, fps_list):
                # Check if the current video matches the specified ID
                if video == id:
                    counter = 0
                    # Iterate through each start time
                    for s in start:
                        # Check if the start time matches the specified start time
                        if int(start_) == s:
                            # Return relevant information once found
                            return (video, s, end[counter], time_of_day_[counter], city, state,
                                    country, (gdp/population), population, population_country,
                                    traffic_mortality, continent, literacy_rate, avg_height, iso_country, fps)
                        counter += 1

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
        """Calculates the total number of videos in the mapping file."""
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
        """Calculates the number of unique countries from a DataFrame.

        Args:
            df (DataFrame): A DataFrame containing the CSV data.

        Returns:
            tuple: A set of unique countries and the total count of unique countries.
        """
        # Extract unique countries from the 'country' column
        unique_countries = set(df[value].unique())

        return unique_countries, len(unique_countries)

    @staticmethod
    def format_city_state(city_state):
        """
        Formats a city_state string or a list of strings in the format 'City_State'.
        If the state is 'unknown', only the city is returned.
        Handles cases where the format is incorrect or missing the '_'.

        Args:
            city_state (str or list): A single string or list of strings in the format 'City_State'.

        Returns:
            str or list: A formatted string or list of formatted strings in the format 'City, State' or 'City'.
        """
        if isinstance(city_state, str):  # If input is a single string
            if "_" in city_state:
                city, state = city_state.split("_", 1)
                return f"{city}, {state}" if state.lower() != "unknown" else city
            else:
                return city_state  # Return as-is if no '_' in string
        elif isinstance(city_state, list):  # If input is a list
            formatted_list = []
            for cs in city_state:
                if "_" in cs:
                    city, state = cs.split("_", 1)
                    if state.lower() != "unknown":
                        formatted_list.append(f"{city}, {state}")
                    else:
                        formatted_list.append(city)
                else:
                    formatted_list.append(cs)  # Append as-is if no '_'
            return formatted_list
        else:
            raise TypeError("Input must be a string or a list of strings.")

    @staticmethod
    def get_value(df, column_name1, column_value1, column_name2, column_value2, target_column):
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

    @staticmethod
    def get_coordinates(city_country, city_coordinates):
        """Get city coordinates either from the pickle file or geocode them."""
        if city_country in city_coordinates:
            return city_coordinates[city_country]
        else:
            # Generate a unique user agent with the current date and time
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            user_agent = f"my_geocoding_script_{current_time}"

            # Create a geolocator with the dynamically generated user_agent
            geolocator = Nominatim(user_agent=user_agent)

            try:
                # Attempt to geocode the city and country with a longer timeout
                location = geolocator.geocode(city_country, timeout=2)  # type: ignore # Set a 2-second timeout

                if location:
                    city_coordinates[city_country] = (location.latitude, location.longitude)  # type: ignore
                    return location.latitude, location.longitude  # type: ignore
                else:
                    logger.error(f"Failed to geocode {city_country}")
                    return None, None  # Return None if city is not found

            except GeocoderTimedOut:
                logger.error(f"Geocoding timed out for {city_country}. Retrying...")

    @staticmethod
    def get_world_plot(df_mapping):
        logger.info("Generating map.")
        cities = df_mapping["city"]
        states = df_mapping["state"]
        countries = df_mapping["country"]
        gdp_city = df_mapping["gdp_city_(billion_US)"]
        population_city = df_mapping["population_city"]
        population_country = df_mapping["population_country"]
        traffic_mortality_rate = df_mapping["traffic_mortality"]
        continent = df_mapping["continent"]
        literacy_rate = df_mapping["literacy_rate"]
        avg_height = df_mapping["avg_height"]
        gini = df_mapping["gini"]
        traffic_index = df_mapping["traffic_index"]

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

        # Load city coordinates from the pickle file if it exists
        if os.path.exists(pickle_file_coordinates):
            with open(pickle_file_coordinates, 'rb') as f:
                city_coordinates = pickle.load(f)
        else:
            city_coordinates = {}

        # Process each city and its corresponding country
        city_coords = []
        for i, (city, state) in enumerate(tqdm(zip(cities, states), total=len(cities))):
            if state and str(state).lower() != 'nan':
                city_country = f"{city}, {state}, {countries[i]}"  # Combine city, state and country
            else:
                city_country = f"{city}, {countries[i]}"  # Combine city and country
                state = 'N/A'
            lat, lon = Analysis.get_coordinates(city_country, city_coordinates)  # type: ignore
            if lat and lon:
                city_coords.append({
                    'City': city,
                    'State': state,
                    'Country': countries[i],
                    'Continent': continent[i],
                    'lat': lat,
                    'lon': lon,
                    'GDP (Billion USD)': gdp_city[i],
                    'City population (thousands)': population_city[i] / 1000.0,
                    'Country population (thousands)': population_country[i] / 1000.0,
                    'Traffic mortality rate (per 100,000)': traffic_mortality_rate[i],
                    'Literacy rate': literacy_rate[i],
                    'Average height (cm)': avg_height[i],
                    'Gini coefficient': gini[i],
                    'Traffic index': traffic_index[i],
                })

        # Save the updated city coordinates back to the pickle file
        with open(pickle_file_coordinates, 'wb') as f:
            pickle.dump(city_coordinates, f)

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
        Analysis.save_plotly_figure(fig, "world_map", save_final=True)

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

        return len(crossed_ids), crossed_ids

    @staticmethod
    def time_to_cross(dataframe, ids, video_id):
        """Calculates the time taken for each object with specified IDs to cross the road.

        Args:
            dataframe (DataFrame): The DataFrame (csv file) containing object data.
            ids (list): A list of unique IDs of objects which are crossing the road.

        Returns:
            dict: A dictionary where keys are object IDs and values are the time taken for
            each object to cross the road, in seconds.
        """
        result = Analysis.find_values_with_video_id(df_mapping, video_id)

        # Check if the result is None (i.e., no matching data was found)
        if result is not None:
            # Unpack the result since it's not None
            (video, start, end, time_of_day, city, state, country, gdp_, population, population_country,
             traffic_mortality_, continent, literacy_rate, avg_height, iso_country, fps) = result

        # Initialize an empty dictionary to store time taken for each object to cross
        var = {}

        # Iterate through each object ID
        for id in ids:
            # Find the minimum and maximum x-coordinates for the object's movement
            x_min = dataframe[dataframe["Unique Id"] == id]["X-center"].min()
            x_max = dataframe[dataframe["Unique Id"] == id]["X-center"].max()

            # Get a sorted group of entries for the current object ID
            sorted_grp = dataframe[dataframe["Unique Id"] == id]

            # Find the index of the minimum and maximum x-coordinates
            x_min_index = sorted_grp[sorted_grp['X-center'] == x_min].index[0]
            x_max_index = sorted_grp[sorted_grp['X-center'] == x_max].index[0]

            # Initialize count and flag variables
            count, flag = 0, 0

            # Determine direction of movement and calculate time taken accordingly
            if x_min_index < x_max_index:
                for value in sorted_grp['X-center']:
                    if value == x_min:
                        flag = 1
                    if flag == 1:
                        count += 1
                        if value == x_max:
                            # Calculate time taken for crossing and store in dictionary
                            var[id] = count/fps
                            break

            else:
                for value in sorted_grp['X-center']:
                    if value == x_max:
                        flag = 1
                    if flag == 1:
                        count += 1
                        if value == x_min:
                            # Calculate time taken for crossing and store in dictionary
                            var[id] = count / fps
                            break

        return var

    @staticmethod
    def calculate_cell_phones(df_mapping, dfs):
        """Plots the relationship between average cell phone usage per person detected vs. traffic mortality.

        Args:
            df_mapping (DataFrame): DataFrame containing mapping information.
            dfs (dict): Dictionary of DataFrames containing video data.
        """
        info, no_person, total_time = {}, {}, {}
        time_ = []
        for key, value in dfs.items():
            # Extract relevant information using the find_values function
            result = Analysis.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                # Unpack the result since it's not None
                (video, start, end, time_of_day, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso_country, fps) = result

                # Count the number of mobile objects in the video
                mobile_ids = Analysis.count_object(value, 67)

                # Calculate the duration of the video
                duration = end - start
                time_.append(duration)

                # Count the number of people in the video
                num_person = Analysis.count_object(value, 0)

                # Extract the time of day
                condition = time_of_day

                # Calculate average cell phones detected per person
                if num_person == 0 or mobile_ids == 0:
                    continue

                # Update the information dictionary
                if f"{city}_{state}_{condition}" in info:
                    previous_value = info[f"{city}_{state}_{condition}"]
                    # Extracting the old number of detected mobiles
                    previous_value = previous_value * no_person[f"{city}_{state}_{condition}"] * total_time[
                        f"{city}_{state}_{condition}"] / 1000 / 60

                    # Summing up the previous value and the new value
                    total_value = previous_value + mobile_ids
                    no_person[f"{city}_{state}_{condition}"] += num_person
                    total_time[f"{city}_{state}_{condition}"] += duration

                    # Normalising with respect to total person detected and time
                    info[f"{city}_{state}_{condition}"] = (((total_value * 60) / total_time[
                        f"{city}_{state}_{condition}"]) / no_person[f"{city}_{state}_{condition}"]) * 1000
                    continue  # Skip saving the variable in plotting variables
                else:
                    no_person[f"{city}_{state}_{condition}"] = num_person
                    total_time[f"{city}_{state}_{condition}"] = duration

                    """Normalising the detection with respect to time and numvber of person in the video.
                    Multiplied by 1000 to increase the value to look better in plotting."""

                    avg_cell_phone = (((mobile_ids * 60) / time_[-1]) / num_person) * 1000
                    info[f"{city}_{state}_{condition}"] = avg_cell_phone

            else:
                # Handle the case where no data was found for the given key
                logger.error(f"No matching data found for key: {key}")

        return info

    @staticmethod
    def calculate_traffic(df_mapping, dfs, person=0, bicycle=0, motorcycle=0, car=0, bus=0, truck=0):
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

        info = {}
        time_ = []

        # Iterate through each video DataFrame
        for key, value in dfs.items():
            result = Analysis.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                # Unpack the result since it's not None
                (video, start, end, time_of_day, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso_country, fps) = result

                # Calculate the duration of the video
                duration = end - start
                time_.append(duration)

                dataframe = value

                # Extract the time of day
                condition = time_of_day

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

                # Calculate normalized vehicle detection rate
                new_value = ((len(vehicle_ids)/time_[-1]) * 60)

                # Update the information dictionary
                if f"{city}_{state}_{condition}" in info:
                    previous_value = info[f"{city}_{state}_{condition}"]
                    info[f"{city}_{state}_{condition}"] = (previous_value + new_value) / 2
                    continue
                else:
                    info[f"{city}_{state}_{condition}"] = new_value

        return info

    @staticmethod
    def calculate_speed_of_crossing(df_mapping, dfs, data, person_id=0):
        speed_dict = {}
        time_ = []
        # Create a dictionary to store country information for each city
        city_country_map_ = {}
        # Iterate over each video data
        for key, df in data.items():
            if df == {}:  # Skip if there is no data
                continue
            result = Analysis.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                (_, start, end, condition, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso_country, fps) = result

                value = dfs.get(key)

                # Store the country associated with each city
                city_country_map_[f'{city}_{state}'] = iso_country

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

                    # Taken from https://doi.org/10.1177/0361198106198200104
                    if speed_ > 1.2:  # Exclude outlier speeds
                        continue
                    if f'{city}_{state}_{condition}' in speed_dict:
                        speed_dict[f'{city}_{state}_{condition}'].append(speed_)
                    else:
                        speed_dict[f'{city}_{state}_{condition}'] = [speed_]

        return speed_dict

    @staticmethod
    def avg_speed_of_crossing(df_mapping, dfs, data):

        speed_array = Analysis.calculate_speed_of_crossing(df_mapping, dfs, data)
        avg_speed = {key: sum(values) / len(values) for key, values in speed_array.items()}

        return avg_speed

    @staticmethod
    def time_to_start_cross(df_mapping, dfs, data, person_id=0):
        time_dict = {}
        for key, df in dfs.items():
            data_cross = {}
            crossed_ids = df[(df["YOLO_id"] == person_id)]

            # Extract relevant information using the find_values function
            result = Analysis.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                (_, start, end, condition, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso_country, fps) = result

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

                if len(data_cross) == 0:
                    continue

                if f'{city}_{state}_{condition}' in time_dict:
                    time_dict[f'{city}_{state}_{condition}'].extend([value / (fps/10) for key,
                                                                     value in data_cross.items()])
                else:
                    time_dict[f'{city}_{state}_{condition}'] = [value / (fps/10) for key, value in data_cross.items()]

        return time_dict

    @staticmethod
    def avg_time_to_start_cross(df_mapping, dfs, data):

        time_array = Analysis.time_to_start_cross(df_mapping, dfs, data)
        avg_time = {key: sum(values) / len(values) for key, values in time_array.items()}

        return avg_time

    @staticmethod
    def traffic_signs(df_mapping, dfs):
        """Plots traffic safety vs traffic mortality.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
        """
        info, duration_ = {}, {}  # Dictionaries to store information and duration

        # Loop through each video data
        for key, value in dfs.items():

            # Extract relevant information using the find_values function
            result = Analysis.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                (_, start, end, time_of_day, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso_country, fps) = result

                dataframe = value

                duration = end - start
                condition = time_of_day

                # Filter dataframe for traffic instruments (YOLO_id 9 and 11)
                instrument = dataframe[(dataframe["YOLO_id"] == 9) | (dataframe["YOLO_id"] == 11)]

                instrument_ids = instrument["Unique Id"].unique()

                # Skip if there are no instrument ids
                if instrument_ids is None:
                    continue

                # Calculate count of traffic instruments detected per minute
                count_ = ((len(instrument_ids)/duration) * 60)

                # Update info dictionary with count normalized by duration
                if f'{city}_{state}_{condition}' in info:
                    old_count = info[f'{city}_{state}_{condition}']
                    new_count = (old_count * duration_.get(f'{city}_{state}_{condition}', 0)) + count_
                    if f'{city}_{state}_{condition}' in duration_:
                        duration_[f'{city}_{state}_{condition}'] = duration_.get(f'{city}_{state}_{condition}',
                                                                                 0) + count
                    else:
                        duration_[f'{city}_{state}_{condition}'] = count
                    info[f'{city}_{state}_{condition}'] = new_count / duration_.get(f'{city}_{state}_{condition}', 0)
                    continue
                else:
                    info[f'{city}_{state}_{condition}'] = count_

        return info

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
        for key, df in data.items():
            counter_exists, counter_nt_exists = 0, 0

            # Extract relevant information using the find_values function
            result = Analysis.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:

                (_, start, end, time_of_day, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso_country, fps) = result

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

                counter_1[f'{city}_{state}_{condition}'] = counter_1.get(f'{city}_{state}_{condition}',
                                                                         0) + var_exist[key]
                counter_2[f'{city}_{state}_{condition}'] = counter_2.get(f'{city}_{state}_{condition}',
                                                                         0) + var_nt_exist[key]

                if (counter_1[f'{city}_{state}_{condition}'] + counter_2[f'{city}_{state}_{condition}']) == 0:
                    # Gives an error of division by 0
                    continue
                else:
                    if f'{city}_{state}_{condition}' in ratio:
                        ratio[f'{city}_{state}_{condition}'] = ((counter_2[f'{city}_{state}_{condition}'] * 100) /
                                                                (counter_1[f'{city}_{state}_{condition}'] +
                                                                counter_2[f'{city}_{state}_{condition}']))
                        continue
                    # If already present, the array below will be filled multiple times
                    else:
                        ratio[f'{city}_{state}_{condition}'] = ((counter_2[f'{city}_{state}_{condition}'] * 100) /
                                                                (counter_1[f'{city}_{state}_{condition}'] +
                                                                counter_2[f'{city}_{state}_{condition}']))
        return ratio

    @staticmethod
    def pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping):
        final = {}
        count = {key: value['count'] for key, value in pedestrian_crossing_count.items()}

        for key, df in count.items():
            result = Analysis.find_values_with_video_id(df_mapping, key)

            if result is not None:
                (_, start, end, time_of_day, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso_country, fps) = result

                # Create the city_time_key (city + time_of_day)
                city_time_key = f'{city}_{state}_{time_of_day}'

                # Add the count to the corresponding city_time_key in the final dict
                if city_time_key in final:
                    final[city_time_key] += count[key]  # Add the current count to the existing sum
                else:
                    final[city_time_key] = count[key]

        return final

# Plotting functions:

    @staticmethod
    def speed_and_time_to_start_cross(df_mapping):
        final_dict = {}
        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[-2]
        avg_time = data_tuple[-3]

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
            city, state, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = Analysis.get_value(df_mapping, "city", city, "state", state, "country")
            iso_code = Analysis.get_value(df_mapping, "city", city, "state", state, "ISO_country")
            if country or iso_code is not None:

                # Initialize the city's dictionary if not already present
                if f'{city}_{state}' not in final_dict:
                    final_dict[f"{city}_{state}"] = {
                        "speed_0": None, "speed_1": None, "time_0": None, "time_1": None,
                        "country": country, "iso": iso_code}

                # Populate the corresponding speed and time based on the condition
                final_dict[f"{city}_{state}"][f"speed_{condition}"] = speed
                if f'{city}_{state}_{condition}' in avg_time:
                    final_dict[f"{city}_{state}"][f"time_{condition}"] = avg_time[f'{city}_{state}_{condition}']

        # Extract all valid speed_0 and speed_1 values along with their corresponding cities
        diff_speed_values = [(f'{city}', abs(data['speed_0'] - data['speed_1']))
                             for city, data in final_dict.items()
                             if data['speed_0'] is not None and data['speed_1'] is not None]

        if diff_speed_values:
            # Sort the list by the absolute difference and get the top 5 and bottom 5
            sorted_diff_speed_values = sorted(diff_speed_values, key=lambda x: x[1], reverse=True)

            top_5_max_speed = sorted_diff_speed_values[:5]  # Top 5 maximum differences
            top_5_min_speed = sorted_diff_speed_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("\n\nTop 5 cities with max |speed_0 - speed_1| differences:")
            for city, diff in top_5_max_speed:
                logger.info(f"{Analysis.format_city_state(city)}: {diff}")

            logger.info("\n\nTop 5 cities with min |speed_0 - speed_1| differences:")
            for city, diff in top_5_min_speed:
                logger.info(f"{Analysis.format_city_state(city)}: {diff}")
        else:
            logger.info("\n\nNo valid speed_0 and speed_1 values found for comparison.")

        # Extract all valid time_0 and time_1 values along with their corresponding cities
        diff_time_values = [(city, abs(data['time_0'] - data['time_1']))
                            for city, data in final_dict.items()
                            if data['time_0'] is not None and data['time_1'] is not None]

        if diff_time_values:
            sorted_diff_time_values = sorted(diff_time_values, key=lambda x: x[1], reverse=True)

            top_5_max = sorted_diff_time_values[:5]  # Top 5 maximum differences
            top_5_min = sorted_diff_time_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("\n\nTop 5 cities with max |time_0 - time_1| differences:")
            for city, diff in top_5_max:
                logger.info(f"{Analysis.format_city_state(city)}: {diff}")

            logger.info("\n\nTop 5 cities with min |time_0 - time_1| differences:")
            for city, diff in top_5_min:
                logger.info(f"{Analysis.format_city_state(city)}: {diff}")
        else:
            logger.info("\n\nNo valid time_0 and time_1 values found for comparison.")

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

            logger.info(f"\n\nCity with max speed at day: {Analysis.format_city_state(max_speed_city_0)} with speed is {max_speed_value_0}m/s")  # noqa:E501
            logger.info(f"\nCity with min speed at day: {Analysis.format_city_state(min_speed_city_0)} with speed is {min_speed_value_0}m/s")  # noqa:E501

        if filtered_dict_s_1:
            max_speed_city_1 = max(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            min_speed_city_1 = min(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            max_speed_value_1 = filtered_dict_s_1[max_speed_city_1]["speed_1"]
            min_speed_value_1 = filtered_dict_s_1[min_speed_city_1]["speed_1"]

            logger.info(f"\n\nCity with max speed at night: {Analysis.format_city_state(max_speed_city_1)} with speed is {max_speed_value_1}m/s")  # noqa:E501
            logger.info(f"\nCity with min speed at night: {Analysis.format_city_state(min_speed_city_1)} with speed is {min_speed_value_1}m/s")  # noqa:E501

        # Find city with max and min time_0 and time_1
        if filtered_dict_t_0:
            max_time_city_0 = max(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            min_time_city_0 = min(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            max_time_value_0 = filtered_dict_t_0[max_time_city_0]["time_0"]
            min_time_value_0 = filtered_dict_t_0[min_time_city_0]["time_0"]

            logger.info(f"\n\nCity with max time at day: {Analysis.format_city_state(max_time_city_0)} with time is {max_time_value_0}s")  # noqa:E501
            logger.info(f"\nCity with min time at day: {Analysis.format_city_state(min_time_city_0)} with time_0 is {min_time_value_0}s")  # noqa:E501

        if filtered_dict_t_1:
            max_time_city_1 = max(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            min_time_city_1 = min(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            max_time_value_1 = filtered_dict_t_1[max_time_city_1]["time_1"]
            min_time_value_1 = filtered_dict_t_1[min_time_city_1]["time_1"]

            logger.info(f"\n\nCity with max time at night: {Analysis.format_city_state(max_time_city_1)} with time is {max_time_value_1}s")  # noqa:E501
            logger.info(f"\nCity with min time at night: {Analysis.format_city_state(min_time_city_1)} with time_1 is {min_time_value_1}s")  # noqa:E501

        # Extract valid speed and time values and calculate statistics
        speed_0_values = [data['speed_0'] for data in final_dict.values() if pd.notna(data['speed_0'])]
        speed_1_values = [data['speed_1'] for data in final_dict.values() if pd.notna(data['speed_1'])]
        time_0_values = [data['time_0'] for data in final_dict.values() if pd.notna(data['time_0'])]
        time_1_values = [data['time_1'] for data in final_dict.values() if pd.notna(data['time_1'])]

        if speed_0_values:
            mean_speed_0 = statistics.mean(speed_0_values)
            sd_speed_0 = statistics.stdev(speed_0_values) if len(speed_0_values) > 1 else 0
            logger.info(f"\n\nMean of speed during day time: {mean_speed_0}")
            logger.info(f"\nStandard deviation of speed during day time: {sd_speed_0}")
        else:
            logger.error("No valid speed during day time values found.")

        if speed_1_values:
            mean_speed_1 = statistics.mean(speed_1_values)
            sd_speed_1 = statistics.stdev(speed_1_values) if len(speed_1_values) > 1 else 0
            logger.info(f"\n\nMean of speed during night time: {mean_speed_1}")
            logger.info(f"\nStandard deviation of speed during night time: {sd_speed_1}")
        else:
            logger.error("No valid speed during night time values found.")

        if time_0_values:
            mean_time_0 = statistics.mean(time_0_values)
            sd_time_0 = statistics.stdev(time_0_values) if len(time_0_values) > 1 else 0
            logger.info(f"\n\nMean of time during day time: {mean_time_0}")
            logger.info(f"\nStandard deviation of time during day time: {sd_time_0}")
        else:
            logger.error("No valid time during day time values found.")

        if time_1_values:
            mean_time_1 = statistics.mean(time_1_values)
            sd_time_1 = statistics.stdev(time_1_values) if len(time_1_values) > 1 else 0
            logger.info(f"\n\nMean of time during night time: {mean_time_1}")
            logger.info(f"\nStandard deviation of time during night time: {sd_time_1}")
        else:
            logger.error("No valid time during night time values found.")

        # Extract city, condition, and count_ from the info dictionary
        cities, conditions_, counts = [], [], []
        for key, value in avg_time.items():
            city, state, condition = key.split('_')
            cities.append(f'{city}_{state}')
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

        # Make sure the lengths match
        # assert len(cities) == len(day_avg_speed) == len(night_avg_speed) == len(
        #     day_time_dict) == len(night_time_dict), "Lengths of lists don't match!"

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups

        fig = make_subplots(
            rows=num_cities_per_col * 2, cols=2,  # Two columns
            vertical_spacing=0,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[2.0] * (num_cities_per_col * 2),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city_new, state = city.split('_')
            iso_code = Analysis.get_value(df_mapping, "city", city_new, "state", state, "ISO_country")
            city = Analysis.format_city_state(city)  # type: ignore
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
            city_new, state = city.split('_')
            iso_code = Analysis.get_value(df_mapping, "city", city_new, "state", state, "ISO_country")
            city = Analysis.format_city_state(city)  # type: ignore
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
            title=dict(text="Speed of crossing the road (m/s)", font=dict(size=40)),
            tickfont=dict(size=40),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=1
        )
        fig.update_xaxes(
            title=dict(text="Speed of crossing the road (m/s)", font=dict(size=40)),
            tickfont=dict(size=40),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=2
        )
        fig.update_xaxes(
            title=dict(text="Time taken to start crossing the road (s)", font=dict(size=40)),
            tickfont=dict(size=40),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=num_cities_per_col * 2,
            col=1
        )

        fig.update_xaxes(
            title=dict(text="Time taken to start crossing the road (s)", font=dict(size=40)),
            tickfont=dict(size=40),
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
            height=7016, width=4960, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Set the x-axis range to cover the values you want in x_grid_values
        x_grid_values = [2, 4, 6, 8, 10, 12, 14, 16, 18]

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
            {"name": "Pedestrian crossing speed during daytime", "color": bar_colour_1},
            {"name": "Pedestrian crossing speed during night time", "color": bar_colour_2},
            {"name": "Pedestrian crossing decision time during daytime", "color": bar_colour_3},
            {"name": "Pedestrian crossing decision time during night time", "color": bar_colour_4},
        ]

        # Add vertical legends with the positions you will provide
        x_legend_position = 0.81  # Position close to the left edge
        y_legend_start_bottom = 0.98  # Lower position to the bottom left corner

        # Add the vertical legends at the top and bottom
        Analysis.add_vertical_legend_annotations(fig, legend_items, x_position=x_legend_position + 0.005,
                                                 y_start=y_legend_start_bottom + 0.02, spacing=0.007, font_size=40)

        # Add a box around the legend
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=x_legend_position,  # Adjust x0 to control the left edge of the box
            y0=y_legend_start_bottom + 0.02,  # Adjust y0 to control the top of the box
            x1=x_legend_position + 0.19,  # Adjust x1 to control the right edge of the box
            # Adjust y1 to control the bottom of the box
            y1=y_legend_start_bottom - len(legend_items) * 0.012 + 0.0395,
            line=dict(color="black", width=2),  # Black border for the box
            fillcolor="rgba(255,255,255,0.7)"  # White fill with transparency
        )

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
        font_size = 20  # Font size for visibility

        # Initialize variables for dynamic y positioning for both columns
        current_row_left = 1  # Start from the first row for the left column
        current_row_right = 1  # Start from the first row for the right column
        y_position_map_left = {}  # Store y positions for each country (left column)
        y_position_map_right = {}  # Store y positions for each country (right column)

        # Calculate the y positions dynamically for the left column
        for city in left_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_left:  # Add the country label once per country
                y_position_map_left[country] = 1 - (current_row_left - 1) / ((len(left_column_cities)-0.6) * 2)

            current_row_left += 2  # Increment the row for each city (speed and time take two rows)

        # Calculate the y positions dynamically for the right column
        for city in right_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_right:  # Add the country label once per country
                y_position_map_right[country] = 1 - (current_row_right - 1) / ((len(right_column_cities)+0.8) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        # Add annotations for country names dynamically for the left column
        for country, y_position in y_position_map_left.items():
            iso2 = Analysis.iso3_to_iso2(country)
            country = country + Analysis.iso2_to_flag(iso2)
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
            iso2 = Analysis.iso3_to_iso2(country)
            country = country + Analysis.iso2_to_flag(iso2)
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
            tickfont=dict(size=20, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=150, b=180))
        Analysis.save_plotly_figure(fig, "consolidated", height=7016, width=4960, scale=3, save_final=True)

    @staticmethod
    def time_to_start_crossing_vs_literacy(df_mapping, need_annotations=True):
        literacy, continents, gdp = [], [], []  # Lists for literacy, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        time = data_tuple[-3]

        for key, value in time.items():
            city, state, condition = key.split('_')
            if need_annotations:
                cities.append(f'{city}_{state}')
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            literacy.append(float(Analysis.get_value(df_mapping, "city", city,
                                                     "state", state, "literacy_rate")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))
            population_city = float(Analysis.get_value(df_mapping, "city", city,
                                                       "state", state, "population_city"))  # type: ignore

            gdp.append(float(Analysis.get_value(df_mapping, "city",
                                                city, "state", state,
                                                "gdp_city_(billion_US)"))/population_city)  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=literacy, y=time, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="time_to_start_crossing_vs_literacy",
                                   x_label="Literacy rate in the country (in percentage)",
                                   y_label="Time taken by pedestrian to start crossing the road (in s)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def time_to_start_crossing_vs_traffic_mortality(df_mapping, need_annotations=True):
        traffic_deaths, continents, gdp = [], [], []  # Lists for traffic related deaths, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []

        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        time = data_tuple[-3]

        for key, value in time.items():
            city, state, condition = key.split('_')
            if need_annotations:
                cities.append(f'{city}_{state}')
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            traffic_deaths.append(float(Analysis.get_value(df_mapping, "city", city,
                                                           "state", state, "literacy_rate")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))
            population_city = float(Analysis.get_value(df_mapping, "city", city,
                                                       "state", state, "population_city"))  # type: ignore
            gdp.append(float(Analysis.get_value(df_mapping,
                                                "city", city, "state", state,
                                                "gdp_city_(billion_US)"))/population_city)  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=traffic_deaths, y=time, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="time_to_start_crossing_vs_traffic_mortality",
                                   x_label="Time to start crossing (in seconds)",
                                   y_label="Traffic mortality",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def traffic_safety_vs_literacy(df_mapping, need_annotations=True):
        """Plots traffic safety vs literacy.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
        """
        literacy, continents, gdp = [], [], []  # Lists for literacy, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        info = data_tuple[-6]

        for key, value in info.items():
            city, state, condition = key.split('_')
            if need_annotations:
                cities.append(f'{city}_{condition}')
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            literacy.append(float(Analysis.get_value(df_mapping, "city",
                                                     city, "state", state, "traffic_mortality")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))
            population_city = float(Analysis.get_value(df_mapping, "city", city,
                                                       "state", state, "population_city"))  # type: ignore
            gdp.append(float(Analysis.get_value(df_mapping,
                                                "city", city, "state", state,
                                                "gdp_city_(billion_US)"))/population_city)  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=literacy, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="traffic_safety_equip_vs_literacy",
                                   x_label="Traffic mortality rate (per 100,000 population)",
                                   y_label="Time taken by pedestrian to start crossing the road (in seconds)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def plot_cell_phone_vs_traffic_mortality(df_mapping, need_annotations=True):
        """Plots the relationship between average cell phone usage per person detected vs. traffic mortality.

        Args:
            df_mapping (DataFrame): DataFrame containing mapping information.
            dfs (dict): Dictionary of DataFrames containing video data.
        """
        traffic_deaths, continents, gdp = [], [], []  # Lists for traffic related deaths, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []

        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        info = data_tuple[-7]

        for key, value in info.items():
            city, state, condition = key.split('_')
            if need_annotations:
                cities.append(f'{city}_{state}')
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            traffic_deaths.append(float(Analysis.get_value(df_mapping, "city", city,
                                                           "state", state, "traffic_mortality")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))
            population_city = float(Analysis.get_value(df_mapping, "city", city,
                                                       "state", state, "population_city"))  # type: ignore
            gdp.append(float(Analysis.get_value(df_mapping,
                                                "city", city, "state", state,
                                                "gdp_city_(billion_US)"))/population_city)  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=traffic_deaths, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="cell_phone_vs_traffic_mortality",
                                   x_label="Traffic mortality rate (per 100,000 population)",
                                   y_label="Number of Mobile detected in the video (normalised)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def vehicle_vs_cross_time(df_mapping, need_annotations=True):
        """Plots the relationship between vehicle detection and crossing time.

        Args:
            df_mapping (DataFrame): DataFrame containing mapping information.
            dfs (dict): Dictionary of DataFrames containing video data.
            data (dict): Dictionary containing information about which object is crossing.
        """
        continents, gdp = [], []  # Lists for traffic related deaths, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        time_cal = []

        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        time = data_tuple[-3]
        info = data_tuple[-8]

        for key, value in info.items():
            city, state, condition = key.split('_')
            if need_annotations:
                cities.append(f'{city}_{state}')
            else:
                cities.append("")
            conditions.append(condition)
            time_cal.append(time.get(key))
            counts.append(value)
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))
            population_city = float(Analysis.get_value(df_mapping, "city", city,
                                                       "state", state, "population_city"))  # type: ignore
            gdp.append(float(Analysis.get_value(df_mapping,
                                                "city", city, "state", state,
                                                "gdp_city_(billion_US)"))/population_city)  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=time_cal, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="vehicle_vs_cross_time",
                                   x_label="Traffic mortality rate (per 100,000 population)",
                                   y_label="Number of Mobile detected in the video (normalised)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def gmp_vs_cross_time(df_mapping, need_annotations=True):
        """Plots the relationship between vehicle detection and crossing time.

        Args:
            df_mapping (DataFrame): DataFrame containing mapping information.
            dfs (dict): Dictionary of DataFrames containing video data.
            data (dict): Dictionary containing information about which object is crossing.
        """
        continents = []
        conditions = []  # Lists for conditions, time, and city
        cities, counts, time_cal = [], [], []
        gmp = {}

        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        time = data_tuple[-3]

        for key, value in time.items():
            city, state, condition = key.split('_')
            if need_annotations:
                cities.append(f'{city}_{state}')
            else:
                cities.append("")
            conditions.append(condition)
            time_cal.append(time.get(key))
            counts.append(value)
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))
            population_city = float(Analysis.get_value(df_mapping, "city", city,
                                                       "state", state, "population_city"))  # type: ignore
            gmp[f'{city}_{state}_{condition}'] = float(Analysis.get_value(df_mapping,
                                                                          "city", city, "state", state,
                                                                          "gdp_city_(billion_US)"
                                                                          ))/population_city  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=time_cal, y=gmp, size=[0.01]*len(gmp), color=continents, symbol=conditions,
                                   city=cities, plot_name="gmp_vs_cross_time",
                                   x_label="Time taken for starting to crossing the road",
                                   y_label="Gross Metropolitan product (GMP)",
                                   legend_x=0.97, legend_y=0.96, need_annotations=False)

    @staticmethod
    def gmp_vs_speed(df_mapping, need_annotations=True):
        """Plots the relationship between vehicle detection and crossing time.

        Args:
            df_mapping (DataFrame): DataFrame containing mapping information.
            dfs (dict): Dictionary of DataFrames containing video data.
            data (dict): Dictionary containing information about which object is crossing.
        """
        continents, conditions, cities, counts, time_cal = [], [], [], [], []
        gmp = {}

        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[-2]

        for key, value in avg_speed.items():
            city, state, condition = key.split('_')
            if need_annotations:
                cities.append(f'{city}_{state}')
            else:
                cities.append("")
            conditions.append(condition)
            time_cal.append(avg_speed.get(key))
            counts.append(value)
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))
            population_city = float(Analysis.get_value(df_mapping, "city", city,
                                                       "state", state, "population_city"))  # type: ignore
            gmp[f'{city}_{state}_{condition}'] = float(Analysis.get_value(df_mapping,
                                                                          "city", city, "state", state,
                                                                          "gdp_city_(billion_US)"
                                                                          ))/population_city  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=time_cal, y=gmp, size=[0.01]*len(gmp), color=continents, symbol=conditions,
                                   city=cities, plot_name="gmp_vs_speed",
                                   x_label="Speed of crossing (in m/s)",
                                   y_label="Gross Metropolitan product (GMP)",
                                   legend_x=0.97, legend_y=0.96, need_annotations=False)

    @staticmethod
    def speed_vs_cross_time(df_mapping, need_annotations=True):
        continents, conditions, cities, counts, time_cal = [], [], [], [], []
        y_axis = {}

        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[-2]
        time = data_tuple[-3]

        for key, value in avg_speed.items():
            if key in time:
                city, state, condition = key.split('_')
                if need_annotations:
                    cities.append(f'{city}_{state}')
                else:
                    cities.append("")
                conditions.append(condition)
                time_cal.append(avg_speed.get(key))
                y_axis[f'{city}_{state}_{condition}'] = time.get(key)
                counts.append(value)
                continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=time_cal, y=y_axis, size=[0.01]*len(y_axis), color=continents, symbol=conditions,
                                   city=cities, plot_name="speed_vs_time",
                                   x_label="Speed of crossing (in m/s)",
                                   y_label="Time taken for starting to crossing the road",
                                   legend_x=0.97, legend_y=0.96, need_annotations=False)

    @staticmethod
    def speed_vs_cross_time_avg(df_mapping, need_annotations=True):
        continents, conditions, cities, counts, time_cal = [], [], [], [], []
        y_axis = {}

        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[-2]
        time = data_tuple[-3]

        # Group data by city and state
        grouped_data = defaultdict(lambda: {"avg_speed": [], "time": []})

        for key, value in avg_speed.items():
            if key in time:
                city, state, condition = key.split('_')
                grouped_key = f"{city}_{state}"
                grouped_data[grouped_key]["avg_speed"].append(value)
                grouped_data[grouped_key]["time"].append(time[key])

        # Compute averages
        averaged_data = {}
        for key, values in grouped_data.items():
            avg_speed_avg = sum(values["avg_speed"]) / len(values["avg_speed"])
            time_avg = sum(values["time"]) / len(values["time"])
            averaged_data[key] = (avg_speed_avg, time_avg)

        for key, (avg_speed_val, time_val) in averaged_data.items():
            city, state = key.split('_')
            if need_annotations:
                cities.append(key)
            else:
                cities.append("")
            conditions.append(condition)  # Use a placeholder since conditions are averaged
            time_cal.append(avg_speed_val)
            y_axis[key] = time_val
            counts.append(avg_speed_val)  # Alternatively, use a relevant count if needed
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=time_cal, y=y_axis, size=[0.01]*len(y_axis), color=continents, symbol=conditions,
                                   city=cities, plot_name="speed_vs_time_avg",
                                   x_label="Pedestrian crossing speed (in m/s)",
                                   y_label="Pedestrian crossing decision time (in s)",
                                   legend_x=0.83, legend_y=0.96, need_annotations=True)

    @staticmethod
    def traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping, need_annotations=True):
        """Plots traffic mortality rate vs percentage of crossing events without traffic light.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
            data (dict): Dictionary containing pedestrian crossing data.
        """
        traffic_deaths, continents, gdp = [], [], []  # Lists for traffic related deaths, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []

        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        info = data_tuple[-9]

        for key, value in info.items():
            city, state, condition = key.split('_')
            if need_annotations:
                cities.append(f'{city}_{condition}')
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            traffic_deaths.append(float(Analysis.get_value(df_mapping, "city",
                                                           city, "state", state, "literacy_rate")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))
            population_city = float(Analysis.get_value(df_mapping, "city", city,
                                                       "state", state, "population_city"))  # type: ignore
            gdp.append(float(Analysis.get_value(df_mapping,
                                                "city", city, "state", state,
                                                "gdp_city_(billion_US)"))/population_city)  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=traffic_deaths, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="traffic_mortality_vs_crossing_event_wt_traffic_light",
                                   x_label="Literacy rate in the country (in percentage)",
                                   y_label="Percentage of Crossing Event without traffic light (normalised)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def plot_traffic_safety_vs_traffic_mortality(df_mapping, need_annotations=True):
        """Plots traffic safety vs traffic mortality.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
        """
        traffic_deaths, continents, gdp = [], [], []  # Lists for literacy, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        info = data_tuple[-6]

        for key, value in info.items():
            city, state, condition = key.split('_')
            cities.append(f'{city}_{state}')
            conditions.append(condition)
            counts.append(value)
            traffic_deaths.append(float(Analysis.get_value(df_mapping, "city", city,
                                                           "state", state, "traffic_mortality")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "state", state, "continent"))
            population_city = float(Analysis.get_value(df_mapping, "city", city,
                                                       "state", state, "population_city"))  # type: ignore
            gdp.append(float(Analysis.get_value(df_mapping,
                                                "city", city, "state", state,
                                                "gdp_city_(billion_US)"))/population_city)  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=traffic_deaths, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="traffic_safety_equip_vs_traffic_mortality",
                                   x_label="Traffic mortality rate (per 100,000 population)",
                                   y_label="Number of traffic instruments detected (normalised)",
                                   legend_x=0.887, legend_y=0.96)

    @staticmethod
    def plot_speed_to_cross_by_alphabetical_order(df_mapping):
        final_dict = {}
        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[-2]

        # Remove the cities where the values are "nan"
        avg_speed = {key: value for key, value in avg_speed.items() if not (isinstance(
            value, float) and math.isnan(value))}

        # Check if 'speed' is valid
        if avg_speed is None:
            raise ValueError("'speed' returned None, please check the input data or calculations.")

        # Now populate the final_dict with city-wise speed data
        for city_condition, speed in avg_speed.items():
            city, state, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = Analysis.get_value(df_mapping, "city", city, "state", state, "country")
            iso_code = Analysis.get_value(df_mapping, "city", city, "state", state, "ISO_country")
            if country or iso_code is not None:
                # Initialize the city's dictionary if not already present
                if f'{city}_{state}' not in final_dict:
                    final_dict[f'{city}_{state}'] = {"speed_0": None, "speed_1": None,
                                                     "country": country, "iso": iso_code}
                # Populate the corresponding speed based on the condition
                final_dict[f'{city}_{state}'][f"speed_{condition}"] = speed

        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Flatten the city list based on country groupings
        cities_ordered = []
        for country in sorted(country_city_map.keys()):  # Sort countries alphabetically
            cities_in_country = sorted(country_city_map[country])  # Sort cities within each country alphabetically
            cities_ordered.extend(cities_in_country)

        # Prepare data for day and night stacking
        day_avg_speed = [final_dict[city]['speed_0'] for city in cities_ordered]
        night_avg_speed = [final_dict[city]['speed_1'] for city in cities_ordered]

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.001,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            row = i + 1
            if day_avg_speed[i] is not None and night_avg_speed[i] is not None:
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_avg_speed[i] is not None:  # Only day data available
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_avg_speed[i] is not None:  # Only night data available
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            row = i + 1
            idx = num_cities_per_col + i
            if day_avg_speed[idx] is not None and night_avg_speed[idx] is not None:
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_avg_speed[idx] is not None:
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_avg_speed[idx] is not None:
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_avg_speed[i] if day_avg_speed[i] is not None else 0) +
            (night_avg_speed[i] if night_avg_speed[i] is not None else 0)
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
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=False
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=False
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(
            title=dict(text="Speed of crossing the road (m/s)", font=dict(size=40)),
            tickfont=dict(size=40),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=1
        )

        fig.update_xaxes(
            title=dict(text="Speed of crossing the road (m/s)", font=dict(size=40)),
            tickfont=dict(size=40),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
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
            height=3508, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Manually add gridlines using `shapes`
        x_grid_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]  # Define the gridline positions on the x-axis
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
            {"name": "Speed during day", "color": bar_colour_1},
            {"name": "Speed during night", "color": bar_colour_2},
        ]

        # Add vertical legends with the positions you will provide
        x_legend_position = 0.30  # Position close to the left edge
        y_legend_start_bottom = 0.65  # Lower position to the bottom left corner

        # Add the vertical legends at the top and bottom
        Analysis.add_vertical_legend_annotations(fig, legend_items, x_position=x_legend_position,
                                                 y_start=y_legend_start_bottom, spacing=0.02, font_size=40)

        # Add a box around the legend
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=x_legend_position,  # Adjust x0 to control the left edge of the box
            y0=y_legend_start_bottom + 0.01,  # Adjust y0 to control the top of the box
            x1=x_legend_position + 0.195,  # Adjust x1 to control the right edge of the box
            y1=y_legend_start_bottom - len(legend_items) * 0.03 + 0.02,  # Adjust y1 to control the bottom of the box
            line=dict(color="black", width=2),  # Black border for the box
            fillcolor="rgba(255,255,255,0.7)"  # White fill with transparency
        )

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
        font_size = 20  # Font size for visibility

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
                y_position_map_right[country] = 1 - (current_row_right - 1) / ((len(right_column_cities)-0.8) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        # Add annotations for country names dynamically for the left column
        for country, y_position in y_position_map_left.items():
            iso2 = Analysis.iso3_to_iso2(country)
            country = country + Analysis.iso2_to_flag(iso2)
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
            iso2 = Analysis.iso3_to_iso2(country)
            country = country + Analysis.iso2_to_flag(iso2)
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
            tickfont=dict(size=16, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=150, b=180))
        Analysis.save_plotly_figure(fig, "speed_of_crossing_by_alphabetical_order",
                                    width=2400, height=3200, scale=3, save_final=True)

    # Function to add vertical legend annotations
    @staticmethod
    def add_vertical_legend_annotations(fig, legend_items, x_position, y_start, spacing=0.03, font_size=50):
        for i, item in enumerate(legend_items):
            fig.add_annotation(
                x=x_position,  # Use the x_position provided by the user
                y=y_start - i * spacing,  # Adjust vertical position based on index and spacing
                xref='paper', yref='paper', showarrow=False,
                text=f'<span style="color:{item["color"]};">&#9632;</span> {item["name"]}',  # noqa:E501
                font=dict(size=font_size),
                xanchor='left', align='left'  # Ensure the text is left-aligned
            )

    @staticmethod
    def plot_time_to_start_cross_by_alphabetical_order(df_mapping):
        final_dict = {}
        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_time = data_tuple[-3]

        # Check if both 'speed' and 'time' are valid dictionaries
        if avg_time is None:
            raise ValueError("'time' returned None, please check the input data or calculations.")

        # Now populate the final_dict with city-wise data
        for city_condition, speed in avg_time.items():
            city, state, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = Analysis.get_value(df_mapping, "city", city, "state", state, "country")
            iso_code = Analysis.get_value(df_mapping, "city", city, "state", state, "ISO_country")
            if country or iso_code is not None:

                # Initialize the city's dictionary if not already present
                if f'{city}_{state}' not in final_dict:
                    final_dict[f'{city}_{state}'] = {
                        "time_0": None, "time_1": None, "country": country, "iso": iso_code
                    }
                # Populate the corresponding speed and time based on the condition
                final_dict[f"{city}_{state}"][f"speed_{condition}"] = speed
                if f'{city}_{state}_{condition}' in avg_time:
                    final_dict[f"{city}_{state}"][f"time_{condition}"] = avg_time[f'{city}_{state}_{condition}']

        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Flatten the city list based on country groupings
        cities_ordered = []
        for country in sorted(country_city_map.keys()):  # Sort countries alphabetically
            cities_in_country = sorted(country_city_map[country])  # Sort cities within each country alphabetically
            cities_ordered.extend(cities_in_country)

        # Prepare data for day and night stacking
        day_time_dict = [final_dict[city]['time_0'] for city in cities_ordered]
        night_time_dict = [final_dict[city]['time_1'] for city in cities_ordered]

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.001,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            # Row for speed (Day and Night)
            row = i + 1
            if day_time_dict[i] is not None and night_time_dict[i] is not None:
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_1),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_2), text=[''],
                    textposition='auto', showlegend=False), row=row, col=1)

            elif day_time_dict[i] is not None:  # Only day time data available
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_1),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_time_dict[i] is not None:  # Only night time data available
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

        # Similarly for the right column
        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            row = i + 1
            idx = num_cities_per_col + i
            if day_time_dict[idx] is not None and night_time_dict[idx] is not None:
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_1),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_2), text=[''],
                    textposition='inside', showlegend=False), row=row, col=2)

            elif day_time_dict[idx] is not None:  # Only day time data available
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_1),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_time_dict[idx] is not None:  # Only night time data available
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[Analysis.format_city_state(city)], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_time_dict[i] if day_time_dict[i] is not None else 0) +
            (night_time_dict[i] if night_time_dict[i] is not None else 0)
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
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=False
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=False
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(
            title=dict(text="Time taken to start crossing the road (s)", font=dict(size=40)),
            tickfont=dict(size=40),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=1
        )

        fig.update_xaxes(
            title=dict(text="Time taken to start crossing the road (s)", font=dict(size=40)),
            tickfont=dict(size=40),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
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
            height=3508, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Define the legend items
        legend_items = [
            {"name": "Time during day", "color": bar_colour_1},
            {"name": "Time during night", "color": bar_colour_2},
        ]

        # Add vertical legends with the positions you will provide
        x_legend_position = 0.30  # Position close to the left edge
        y_legend_start_bottom = 0.65  # Lower position to the bottom left corner

        # Add the vertical legends at the top and bottom
        Analysis.add_vertical_legend_annotations(fig, legend_items, x_position=x_legend_position,
                                                 y_start=y_legend_start_bottom, spacing=0.02, font_size=40)

        # Manually add gridlines using `shapes`
        x_grid_values = [2, 4, 6, 8, 10, 12, 14, 16, 18]  # Define the gridline positions on the x-axis
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

        # Add a box around the legend
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=x_legend_position,  # Adjust x0 to control the left edge of the box
            y0=y_legend_start_bottom + 0.01,  # Adjust y0 to control the top of the box
            x1=x_legend_position + 0.195,  # Adjust x1 to control the right edge of the box
            y1=y_legend_start_bottom - len(legend_items) * 0.03 + 0.02,  # Adjust y1 to control the bottom of the box
            line=dict(color="black", width=2),  # Black border for the box
            fillcolor="rgba(255,255,255,0.7)"  # White fill with transparency
        )

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
        font_size = 20  # Font size for visibility

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

        # Add annotations for country names dynamically for the left column
        for country, y_position in y_position_map_left.items():
            iso2 = Analysis.iso3_to_iso2(country)
            country = country + Analysis.iso2_to_flag(iso2)
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
            iso2 = Analysis.iso3_to_iso2(country)
            country = country + Analysis.iso2_to_flag(iso2)
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
            tickfont=dict(size=16, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=150, b=180))
        Analysis.save_plotly_figure(fig, "time_to_start_cross", width=2400, height=3200, scale=3, save_final=True)

    @staticmethod
    def plot_speed_to_cross_by_average(df_mapping):
        final_dict = {}
        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[-2]

        # Remove the cities where the values are "nan"
        avg_speed = {key: value for key, value in avg_speed.items() if not (
            isinstance(value, float) and math.isnan(value))}

        # Check if 'speed' is valid
        if avg_speed is None:
            raise ValueError("'speed' returned None, please check the input data or calculations.")

        # Now populate the final_dict with city-wise speed data
        for city_condition, speed in avg_speed.items():
            city, state, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = Analysis.get_value(df_mapping, "city", city, "state", state, "country")
            iso_code = Analysis.get_value(df_mapping, "city", city, "state", state, "ISO_country")
            if country or iso_code is not None:
                # Initialize the city's dictionary if not already present
                if f"{city}_{state}" not in final_dict:
                    final_dict[f"{city}_{state}"] = {"speed_0": None, "speed_1": None,
                                                     "country": country, "iso": iso_code}
                # Populate the corresponding speed based on the condition
                final_dict[f"{city}_{state}"][f"speed_{condition}"] = speed

        # Sort cities by the sum of speed_0 and speed_1 values
        cities_ordered = sorted(
            final_dict.keys(),
            key=lambda city: (final_dict[city]["speed_0"] or 0) + (
                final_dict[city]["speed_1"] or 0), reverse=True)
        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        # Prepare data for day and night stacking
        day_avg_speed = [final_dict[city]['speed_0'] for city in cities_ordered]
        night_avg_speed = [final_dict[city]['speed_1'] for city in cities_ordered]

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.001,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city_new, state = city.split('_')
            iso_code = Analysis.get_value(df_mapping, "city", city_new, "state", state, "ISO_country")
            city = Analysis.format_city_state(
                city) + " " + Analysis.iso2_to_flag(Analysis.iso3_to_iso2(iso_code))  # type: ignore
            row = i + 1
            if day_avg_speed[i] is not None and night_avg_speed[i] is not None:
                value = (day_avg_speed[i] + night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_avg_speed[i] is not None:  # Only day data available
                value = (day_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_avg_speed[i] is not None:  # Only night data available
                value = (night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, state = city.split('_')
            iso_code = Analysis.get_value(df_mapping, "city", city_new, "state", state, "ISO_country")
            city = Analysis.format_city_state(
                city) + " " + Analysis.iso2_to_flag(Analysis.iso3_to_iso2(iso_code))  # type: ignore
            row = i + 1
            idx = num_cities_per_col + i
            if day_avg_speed[idx] is not None and night_avg_speed[idx] is not None:
                value = (day_avg_speed[idx] + night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
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
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_avg_speed[idx] is not None:
                value = (night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_avg_speed[i] if day_avg_speed[i] is not None else 0) +
            (night_avg_speed[i] if night_avg_speed[i] is not None else 0)
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
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=False
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=False
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(title_text="Speed of crossing the road (m/s)", titlefont=dict(size=40),
                         tickfont=dict(size=40), ticks='outside', ticklen=10, tickwidth=2,
                         tickcolor='black', row=1, col=1)
        fig.update_xaxes(title_text="Speed of crossing the road (m/s)", titlefont=dict(size=40),
                         tickfont=dict(size=40), ticks='outside', ticklen=10, tickwidth=2,
                         tickcolor='black', row=1, col=2)

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=3508, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Manually add gridlines using `shapes`
        x_grid_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]  # Define the gridline positions on the x-axis
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
            {"name": "Speed during day", "color": bar_colour_1},
            {"name": "Speed during night", "color": bar_colour_2},
        ]

        # Add vertical legends with the positions you will provide
        x_legend_position = 0.30  # Position close to the left edge
        y_legend_start_bottom = 0.65  # Lower position to the bottom left corner

        # Add the vertical legends at the top and bottom
        Analysis.add_vertical_legend_annotations(fig, legend_items, x_position=x_legend_position,
                                                 y_start=y_legend_start_bottom, spacing=0.02, font_size=40)

        # Add a box around the legend
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=x_legend_position,  # Adjust x0 to control the left edge of the box
            y0=y_legend_start_bottom + 0.01,  # Adjust y0 to control the top of the box
            x1=x_legend_position + 0.195,  # Adjust x1 to control the right edge of the box
            y1=y_legend_start_bottom - len(legend_items) * 0.03 + 0.02,  # Adjust y1 to control the bottom of the box
            line=dict(color="black", width=2),  # Black border for the box
            fillcolor="rgba(255,255,255,0.7)"  # White fill with transparency
        )

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
            tickfont=dict(size=16, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=150, b=180))
        Analysis.save_plotly_figure(fig, "speed_of_crossing_by_avg", width=2400, height=3200, scale=3, save_final=True)

    @staticmethod
    def plot_time_to_start_cross_by_average(df_mapping):
        final_dict = {}
        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_time = data_tuple[-3]

        # Check if both 'speed' and 'time' are valid dictionaries
        if avg_time is None:
            raise ValueError("'time' returned None, please check the input data or calculations.")

        # Now populate the final_dict with city-wise speed data
        for city_condition, time in avg_time.items():
            city, state, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = Analysis.get_value(df_mapping, "city", city, "state", state, "country")
            iso_code = Analysis.get_value(df_mapping, "city", city, "state", state, "ISO_country")
            if country or iso_code is not None:
                # Initialize the city's dictionary if not already present
                if f"{city}_{state}" not in final_dict:
                    final_dict[f"{city}_{state}"] = {"time_0": None,
                                                     "time_1": None, "country": country, "iso": iso_code}
                # Populate the corresponding speed based on the condition
                final_dict[f"{city}_{state}"][f"time_{condition}"] = time

        # Sort cities by the sum of speed_0 and speed_1 values
        cities_ordered = sorted(
            final_dict.keys(),
            key=lambda city: (final_dict[city]["time_0"] or 0) + (final_dict[city]["time_1"] or 0),
            reverse=True
        )
        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        # Prepare data for day and night stacking
        day_time_dict = [final_dict[city]['time_0'] for city in cities_ordered]
        night_time_dict = [final_dict[city]['time_1'] for city in cities_ordered]

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.001,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            # Row for speed (Day and Night)
            city_new, state = city.split('_')
            iso_code = Analysis.get_value(df_mapping, "city", city_new, "state", state, "ISO_country")
            city = Analysis.format_city_state(
                city) + " " + Analysis.iso2_to_flag(Analysis.iso3_to_iso2(iso_code))  # type: ignore
            row = i + 1
            if day_time_dict[i] is not None and night_time_dict[i] is not None:
                value = (day_time_dict[i] + night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    # name=f"{city} time during day", marker=dict(color='#FF6692'),
                    name=f"{city} time during day", marker=dict(color=bar_colour_1),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_2),
                    # name=f"{city} time during night", marker=dict(color='#B6E880'), text=[''],
                    textposition='auto', showlegend=False), row=row, col=1)

            elif day_time_dict[i] is not None:  # Only day time data available
                value = (day_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_1),
                    # name=f"{city} time during day", marker=dict(color='#FF6692'),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_time_dict[i] is not None:  # Only night time data available
                value = (night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_2),
                    # name=f"{city} time during night", marker=dict(color='#B6E880'),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

        # Similarly for the right column
        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, state = city.split('_')
            iso_code = Analysis.get_value(df_mapping, "city", city_new, "state", state, "ISO_country")
            city = Analysis.format_city_state(
                city) + " " + Analysis.iso2_to_flag(Analysis.iso3_to_iso2(iso_code))  # type: ignore
            row = i + 1
            idx = num_cities_per_col + i
            if day_time_dict[idx] is not None and night_time_dict[idx] is not None:
                value = (day_time_dict[idx] + night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_1),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_2), text=[''],
                    textposition='inside', showlegend=False), row=row, col=2)

            elif day_time_dict[idx] is not None:  # Only day time data available
                value = (day_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_1),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_time_dict[idx] is not None:  # Only night time data available
                value = (night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_time_dict[i] if day_time_dict[i] is not None else 0) +
            (night_time_dict[i] if night_time_dict[i] is not None else 0)
            for i in range(len(cities))
        ]) if cities else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column
            if i % 2 == 1:  # Odd rows
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

            # Update x-axis for the right column
            if i % 2 == 1:  # Odd rows
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=True
                )
            else:  # Even rows
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=True
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(title_text="Time taken to start crossing the road (s)", titlefont=dict(size=40),
                         tickfont=dict(size=40), ticks='outside', ticklen=10, tickwidth=2,
                         tickcolor='black', showgrid=True, row=1, col=1)
        fig.update_xaxes(title_text="Time taken to start crossing the road (s)", titlefont=dict(size=40),
                         tickfont=dict(size=40), ticks='outside', ticklen=10, tickwidth=2,
                         tickcolor='black', showgrid=True, row=1, col=2)

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=3508, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Manually add gridlines using `shapes`
        x_grid_values = [2, 4, 6, 8, 10, 12, 14, 16, 18]  # Define the gridline positions on the x-axis
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
            {"name": "Time during day", "color": bar_colour_1},
            {"name": "Time during night", "color": bar_colour_2},
        ]

        # Add vertical legends with the positions you will provide
        x_legend_position = 0.30  # Position close to the left edge
        y_legend_start_bottom = 0.65  # Lower position to the bottom left corner

        # Add the vertical legends at the top and bottom
        Analysis.add_vertical_legend_annotations(fig, legend_items, x_position=x_legend_position,
                                                 y_start=y_legend_start_bottom, spacing=0.02, font_size=40)

        # Add a box around the legend
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=x_legend_position,  # Adjust x0 to control the left edge of the box
            y0=y_legend_start_bottom + 0.01,  # Adjust y0 to control the top of the box
            x1=x_legend_position + 0.195,  # Adjust x1 to control the right edge of the box
            y1=y_legend_start_bottom - len(legend_items) * 0.03 + 0.02,  # Adjust y1 to control the bottom of the box
            line=dict(color="black", width=2),  # Black border for the box
            fillcolor="rgba(255,255,255,0.7)"  # White fill with transparency
        )

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
            tickfont=dict(size=16, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=150, b=180))
        Analysis.save_plotly_figure(fig, "time_to_start_cross_by_avg",
                                    width=2400, height=3200, scale=3, save_final=True)

    @staticmethod
    def correlation_matrix(df_mapping):
        final_dict = {}
        with open(pickle_file_path, 'rb') as file:
            data_tuple = pickle.load(file)

        (ped_cross_city, ped_crossing_count, person_city, bicycle_city, car_city,
         motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city,
         cellphone_city, trf_sign_city, speed_values, time_values, avg_time, avg_speed, df_mapping) = data_tuple[10:]

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
            city, state, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = Analysis.get_value(df_mapping, "city", city, "state", state, "country")
            iso_code = Analysis.get_value(df_mapping, "city", city, "state", state, "ISO_country")
            continent = Analysis.get_value(df_mapping, "city", city, "state", state, "continent")
            population_country = Analysis.get_value(df_mapping, "city", city, "state", state, "population_country")
            gdp_city = Analysis.get_value(df_mapping, "city", city, "state", state, "gdp_city_(billion_US)")
            traffic_mortality = Analysis.get_value(df_mapping, "city", city, "state", state, "traffic_mortality")
            literacy_rate = Analysis.get_value(df_mapping, "city", city, "state", state, "literacy_rate")
            gini = Analysis.get_value(df_mapping, "city", city, "state", state, "gini")
            traffic_index = Analysis.get_value(df_mapping, "city", city, "state", state, "traffic_index")

            if country or iso_code is not None:

                # Initialize the city's dictionary if not already present
                if f'{city}_{state}' not in final_dict:
                    final_dict[f'{city}_{state}'] = {
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
                final_dict[f'{city}_{state}'][f"avg_speed_{condition}"] = speed
                if f'{city}_{state}_{condition}' in avg_time:
                    final_dict[f'{city}_{state}'][f"avg_time_{condition}"] = avg_time.get(
                        f'{city}_{state}_{condition}', None)
                    final_dict[f'{city}_{state}'][f"time_val_{condition}"] = time_values.get(
                        f'{city}_{state}_{condition}', None)
                    final_dict[f'{city}_{state}'][f"speed_val_{condition}"] = speed_values.get(
                        f'{city}_{state}_{condition}', None)
                    final_dict[f'{city}_{state}'][f"time_val_{condition}"] = time_values.get(
                        f'{city}_{state}_{condition}', None)
                    final_dict[f'{city}_{state}'][f"ped_cross_city_{condition}"] = ped_cross_city.get(
                        f'{city}_{state}_{condition}', None)
                    final_dict[f'{city}_{state}'][f"person_city_{condition}"] = person_city.get(
                        f'{city}_{state}_{condition}', 0)
                    final_dict[f'{city}_{state}'][f"bicycle_city_{condition}"] = bicycle_city.get(
                        f'{city}_{state}_{condition}', 0)
                    final_dict[f'{city}_{state}'][f"car_city_{condition}"] = car_city.get(
                        f'{city}_{state}_{condition}', 0)
                    final_dict[f'{city}_{state}'][f"motorcycle_city_{condition}"] = motorcycle_city.get(
                        f'{city}_{state}_{condition}', 0)
                    final_dict[f'{city}_{state}'][f"bus_city_{condition}"] = bus_city.get(
                        f'{city}_{state}_{condition}', 0)
                    final_dict[f'{city}_{state}'][f"truck_city_{condition}"] = truck_city.get(
                        f'{city}_{state}_{condition}', 0)
                    final_dict[f'{city}_{state}'][f"cross_evnt_city_{condition}"] = cross_evnt_city.get(
                        f'{city}_{state}_{condition}', 0)
                    final_dict[f'{city}_{state}'][f"vehicle_city_{condition}"] = vehicle_city.get(
                        f'{city}_{state}_{condition}', None)
                    final_dict[f'{city}_{state}'][f"cellphone_city_{condition}"] = cellphone_city.get(
                        f'{city}_{state}_{condition}', 0)
                    final_dict[f'{city}_{state}'][f"trf_sign_city_{condition}"] = trf_sign_city.get(
                        f'{city}_{state}_{condition}', 0)
                    final_dict[f'{city}_{state}'][f"traffic_mortality_{condition}"] = traffic_mortality
                    final_dict[f'{city}_{state}'][f"literacy_rate_{condition}"] = literacy_rate
                    final_dict[f'{city}_{state}'][f"gini_{condition}"] = gini
                    final_dict[f'{city}_{state}'][f"traffic_index_{condition}"] = traffic_index
                    final_dict[f'{city}_{state}'][f"continent_{condition}"] = continent
                    if gdp_city is not None:
                        final_dict[f'{city}_{state}'][f"gmp_{condition}"] = gdp_city/population_country

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
            'avg_speed_0': 'Speed of', 'avg_speed_1': 'Pedestrian crossing speed',
            'avg_time_0': 'Pedestrian crossing decision time', 'avg_time_1': 'Pedestrian crossing decision time',
            'ped_cross_city_0': 'Pedestrian crossing', 'ped_cross_city_1': 'Pedestrian crossing',
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
        fig = px.imshow(corr_matrix_day, text_auto=".2f",  # Display correlation values on the heatmap
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto")  # Automatically adjust aspect ratio
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        Analysis.save_plotly_figure(fig, "correlation_matrix_heatmap_in_daylight", save_final=True)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_night, text_auto=".2f",  # Display correlation values on the heatmap
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation Matrix Heatmap in night"  # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        Analysis.save_plotly_figure(fig, "correlation_matrix_heatmap_in_night", save_final=True)

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

                if speed_vals:  # Avoid division by zero or empty arrays
                    row_data[f"avg_speed_val_{condition}"] = np.mean(speed_vals)
                else:
                    row_data[f"avg_speed_val_{condition}"] = np.nan  # Handle empty or missing arrays

                if time_vals:
                    row_data[f"avg_time_val_{condition}"] = np.mean(time_vals)
                else:
                    row_data[f"avg_time_val_{condition}"] = np.nan  # Handle empty or missing arrays

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
            'avg_speed_val': 'Pedestrian crossing speed', 'avg_time_val': 'Pedestrian crossing decision time',
            'ped_cross_city': 'Pedestrian crossing', 'person_city': 'Detected persons',
            'bicycle_city': 'Detected bicycles', 'car_city': 'Detected cars',
            'motorcycle_city': 'Detected motorcycles', 'bus_city': 'Detected bus',
            'truck_city': 'Detected truck', 'cross_evnt_city': 'Crossing without traffic light',
            'vehicle_city': 'Detected total number of motor vehicle', 'cellphone_city': 'Detected cellphone',
            'trf_sign_city': 'Detected traffic signs', 'gmp_city': 'GMP',
            'traffic_mortality_city': 'Traffic mortality', 'literacy_rate_city': 'Literacy rate', 'gini': 'Gini coefficient',
            'traffic_index': 'Traffic Index'
            }

        corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_2, index=rename_dict_2)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on the heatmap
                        color_continuous_scale='RdBu',  # Color scale (you can customize this)
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation matrix heatmap averaged" # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        Analysis.save_plotly_figure(fig, "correlation_matrix_heatmap_averaged", save_final=True)

        # Continent Wise

        # Initialize a list to store rows of data (one row per city)
        data_rows = []

        # Assuming `conditions` is a list of conditions you are working with
        conditions = ['0', '1']  # Modify this list to include all conditions you have (e.g., '0', '1', etc.)
        unique_continents = df_mapping['continent'].unique()

        # Iterate over each city and condition
        for city in final_dict:
            # Initialize a dictionary to store the values for the current row
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

                if speed_vals:  # Avoid division by zero or empty arrays
                    row_data[f"avg_speed_val_{condition}"] = np.mean(speed_vals)
                else:
                    row_data[f"avg_speed_val_{condition}"] = np.nan  # Handle empty or missing arrays

                if time_vals:
                    row_data[f"avg_time_val_{condition}"] = np.mean(time_vals)
                else:
                    row_data[f"avg_time_val_{condition}"] = np.nan  # Handle empty or missing arrays

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
                'avg_speed_val': 'Pedestrian crossing speed', 'avg_time_val': 'Pedestrian crossing decision time',
                'ped_cross_city': 'Pedestrian crossing', 'person_city': 'Detected persons',
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
            fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on the heatmap
                            color_continuous_scale='RdBu',  # Color scale (you can customize this)
                            aspect="auto",  # Automatically adjust aspect ratio
                            # title=f"Correlation matrix heatmap {continents}"  # Title of the heatmap
                            )

            fig.update_layout(coloraxis_showscale=False)

            # update font family
            fig.update_layout(font=dict(family=common.get_configs('font_family')))

            # use value from config file
            fig.update_layout(font=dict(size=common.get_configs('font_size')))

            Analysis.save_plotly_figure(fig, f"correlation_matrix_heatmap_{continents}", save_final=True)

    @staticmethod
    def iso2_to_flag(iso2):
        if iso2 is None:
            # Return a placeholder or an empty string if the ISO-2 code is not available
            return "🇽🇰"
        return chr(ord('🇦') + (ord(iso2[0]) - ord('A'))) + chr(ord('🇦') + (ord(iso2[1]) - ord('A')))

    @staticmethod
    def iso3_to_iso2(iso3_code):
        try:
            # Find the country by ISO-3 code
            country = pycountry.countries.get(alpha_3=iso3_code)
            # Return the ISO-2 code
            return country.alpha_2 if country else None
        except AttributeError:
            return None

    @staticmethod
    def find_city_id(df, video_id, start_time):
        for _, row in df.iterrows():
            videos = re.findall(r"[\w-]+", row["videos"])  # convert to list
            start_times = ast.literal_eval(row["start_time"])  # convert to list
            
            if video_id in videos:
                index = videos.index(video_id)  # get the index of the video
                if start_time in start_times[index]:  # check if start_time matches
                    return row["id"]  # return the matching city
    
        return None  # return none if no match is found

    @staticmethod
    def get_duration_segment(df, video_id, start_time):
        """Get duration of segment."""
        for _, row in df.iterrows():
            videos = re.findall(r"[\w-]+", row["videos"])  # convert to list
            start_times = ast.literal_eval(row["start_time"])  # convert to list
            end_times = ast.literal_eval(row["end_time"])  # convert to list
            
            if video_id in videos:
                index = videos.index(video_id)  # get the index of the video
                if start_time in start_times[index]:  # check if start_time matches
                    # find end time that matches the start time
                    index_start = start_times[index].index(start_time)
                    end_time = end_times[index][index_start]
                    return end_time - start_time  # return duration of segment
    
        return None  # return none if no match is found

    @staticmethod
    def scatter(df, x, y, color=None, symbol=None, size=None, text=None, trendline=None, hover_data=None,
                marker_size=None, pretty_text=False, marginal_x='violin', marginal_y='violin', xaxis_title=None,
                yaxis_title=None, xaxis_range=None, yaxis_range=None, name_file=None, save_file=False,
                save_final=False, fig_save_width=1320, fig_save_height=680, font_family=None, font_size=None,
                hover_name=None, legend_title=None):
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

        if text == 'city':
            # use KDTree to check point density
            tree = KDTree(df[[x, y]].values)  # Ensure finite values
            distances, _ = tree.query(df[[x, y]].values, k=2)  # Find nearest neighbor distance

            # define a distance threshold for labeling
            threshold = np.mean(distances[:, 1])

            # only label points that are not too close to others
            df["display_label"] = np.where(distances[:, 1] > threshold, df["city"], "")

            text = "display_label"

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
        fig.update_traces(textfont=dict(size=common.get_configs('font_size')))  # Adjust font size here

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
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'scatter_' + x + '-' + y
            # Final adjustments and display
            fig.update_layout(margin=dict(l=80, r=100, t=150, b=180))
            Analysis.save_plotly_figure(fig, name_file, scatter_plot_flag=False, save_final=True)
        # open it in localhost instead
        else:
            fig.show()

    @staticmethod
    def improve_text_position(x):
        """ it is more efficient if the x values are sorted """
        # fix indentation 
        positions = ['top center', 'bottom center']  # you can add more: left center ...
        return [positions[i % len(positions)] for i in range(len(x))]


# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")

    if os.path.exists(pickle_file_path) and not common.get_configs('always_analyse'):
        # Load the data from the pickle file
        with open(pickle_file_path, 'rb') as file:
            (data, person_counter, bicycle_counter, car_counter, motorcycle_counter,
             bus_counter, truck_counter, cellphone_counter, traffic_light_counter, stop_sign_counter,
             pedestrian_cross_city, pedestrian_crossing_count, person_city, bicycle_city, car_city,
             motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city,
             cellphone_city, traffic_sign_city, speed_values, time_values, avg_time, avg_speed,
             df_mapping) = pickle.load(file)

        logger.info("Loaded analysis results from pickle file.")
    else:
        # Stores the mapping file
        df_mapping = pd.read_csv("mapping.csv")

        pedestrian_crossing_count, data = {}, {}
        person_counter, bicycle_counter, car_counter, motorcycle_counter = 0, 0, 0, 0
        bus_counter, truck_counter, cellphone_counter, traffic_light_counter, stop_sign_counter = 0, 0, 0, 0, 0

        total_duration = Analysis.calculate_total_seconds(df_mapping)
        logger.info(f"Duration of videos in seconds: {total_duration}, in minutes: {total_duration/60:.2f}, in " +
                    f"hours: {total_duration/60/60:.2f}")
        logger.info("Total number of videos: {}", Analysis.calculate_total_videos(df_mapping))
        country, number = Analysis.get_unique_values(df_mapping, "country")
        logger.info("Total number of countries: {}", number)
        city, number = Analysis.get_unique_values(df_mapping, "city")
        logger.info("Total number of cities: {}", number)

        # Stores the content of the csv file in form of {name_time: content}
        dfs = Analysis.read_csv_files(common.get_configs('data'))
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
        # Loop over rows of data
        for key, value in dfs.items():
            # extract information for the csv file from mapping
            video_id, start_index = key.rsplit("_", 1)  # split to extract id and index
            video_city_id = Analysis.find_city_id(df_mapping, video_id, int(start_index))
            video_city = df_mapping.loc[df_mapping["id"] == video_city_id, "city"].values[0]
            video_state = df_mapping.loc[df_mapping["id"] == video_city_id, "state"].values[0]
            video_country = df_mapping.loc[df_mapping["id"] == video_city_id, "country"].values[0]
            logger.info("Analysing data from {} from {}, {}, {}.", key, video_city, video_state, video_country)

            # Get the number of number and unique id of the object crossing the road
            count, ids = Analysis.pedestrian_crossing(dfs[key], 0.45, 0.55, 0)

            # Saving it in a dictionary in: {name_time: count, ids}
            pedestrian_crossing_count[key] = {"count": count, "ids": ids}

            # Saves the time to cross in form {name_time: {id(s): time(s)}}
            data[key] = Analysis.time_to_cross(dfs[key], pedestrian_crossing_count[key]["ids"], key)

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
            time_video = Analysis.get_duration_segment(df_mapping, video_id, int(start_index))
            df_mapping.loc[df_mapping["id"] == video_city_id, "total_time"] += time_video
        logger.info("Calculating aggregated values.")
        speed_values = Analysis.calculate_speed_of_crossing(df_mapping, dfs, data)
        time_values = Analysis.time_to_start_cross(df_mapping, dfs, data)
        avg_speed = Analysis.avg_speed_of_crossing(df_mapping, dfs, data)
        # add to mapping file
        for key, value in avg_speed.items():
            parts = key.split("_")
            city = parts[0]  # First part is always the city
            state = parts[1] if parts[1] != "unknown" else np.nan  # Second part is the state, unless it's "unknown"
            time_of_day = int(parts[2])  # Third part is the time-of-day
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
        avg_time = Analysis.avg_time_to_start_cross(df_mapping, dfs, data)
        # add to mapping file
        for key, value in avg_time.items():
            parts = key.split("_")
            city = parts[0]  # First part is always the city
            state = parts[1] if parts[1] != "unknown" else np.nan  # Second part is the state, unless it's "unknown"
            time_of_day = int(parts[2])  # Third part is the time-of-day
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
        df_mapping["speed_crossing"] = np.where(
            (df_mapping["speed_crossing_day"] > 0) & (df_mapping["speed_crossing_night"] > 0),  
            df_mapping[["speed_crossing_day", "speed_crossing_night"]].mean(axis=1),  
            np.where(
                df_mapping["speed_crossing_day"] > 0, df_mapping["speed_crossing_day"],  
                np.where(df_mapping["speed_crossing_night"] > 0, df_mapping["speed_crossing_night"], np.nan)
            )
        )
        df_mapping["time_crossing"] = np.where(
            (df_mapping["time_crossing_day"] > 0) & (df_mapping["time_crossing_night"] > 0),  
            df_mapping[["time_crossing_day", "time_crossing_night"]].mean(axis=1),  
            np.where(
                df_mapping["time_crossing_day"] > 0, df_mapping["time_crossing_day"],  
                np.where(df_mapping["time_crossing_night"] > 0, df_mapping["time_crossing_night"], np.nan)
            )
        )
        traffic_sign_city = Analysis.traffic_signs(df_mapping, dfs)
        cellphone_city = Analysis.calculate_cell_phones(df_mapping, dfs)
        vehicle_city = Analysis.calculate_traffic(df_mapping, dfs, motorcycle=1, car=1, bus=1, truck=1)
        bicycle_city = Analysis.calculate_traffic(df_mapping, dfs, bicycle=1)
        car_city = Analysis.calculate_traffic(df_mapping, dfs, car=1)
        motorcycle_city = Analysis.calculate_traffic(df_mapping, dfs, motorcycle=1)
        bus_city = Analysis.calculate_traffic(df_mapping, dfs, bus=1)
        truck_city = Analysis.calculate_traffic(df_mapping, dfs, truck=1)
        person_city = Analysis.calculate_traffic(df_mapping, dfs, person=1)
        cross_evnt_city = Analysis.crossing_event_wt_traffic_light(df_mapping, dfs, data)
        pedestrian_cross_city = Analysis.pedestrian_cross_per_city(pedestrian_crossing_count, df_mapping)

        # Save the results to a pickle file
        logger.info("Saving results to a pickle file {}.", pickle_file_path)
        with open(pickle_file_path, 'wb') as file:
            pickle.dump((data, person_counter, bicycle_counter, car_counter, motorcycle_counter,
                         bus_counter, truck_counter, cellphone_counter, traffic_light_counter, stop_sign_counter,
                         pedestrian_cross_city, pedestrian_crossing_count, person_city, bicycle_city, car_city,
                         motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city,
                         cellphone_city, traffic_sign_city, speed_values, time_values, avg_time, avg_speed,
                         df_mapping), file)
        logger.info("Analysis results saved to pickle file.")
    
    # save updated mapping file in output
    df_mapping.to_csv(os.path.join(common.output_dir, "mapping_updated.csv"))

    logger.info("Detected:")
    logger.info(f"person: {person_counter} ; bicycle: {bicycle_counter} ; car: {car_counter}")
    logger.info(f"motorcycle: {motorcycle_counter} ; bus: {bus_counter} ; truck: {truck_counter}")
    logger.info(f"cellphone: {cellphone_counter}; traffic light: {traffic_light_counter}; sign: {stop_sign_counter}")

    logger.info("Producing figures.")

    # Set state to be NA
    # Analysis.get_world_plot(df_mapping)
    Analysis.speed_and_time_to_start_cross(df_mapping)
    # Analysis.time_to_start_crossing_vs_literacy(df_mapping)
    # Analysis.time_to_start_crossing_vs_traffic_mortality(df_mapping)
    # Analysis.traffic_safety_vs_literacy(df_mapping)
    # Analysis.plot_cell_phone_vs_traffic_mortality(df_mapping)
    # Analysis.vehicle_vs_cross_time(df_mapping)
    # Analysis.traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping)
    # Analysis.plot_traffic_safety_vs_traffic_mortality(df_mapping)
    # Analysis.gmp_vs_cross_time(df_mapping)
    # Analysis.gmp_vs_speed(df_mapping)
    # Analysis.speed_vs_cross_time(df_mapping)
    # Analysis.speed_vs_cross_time_avg(df_mapping)
    # Analysis.plot_speed_to_cross_by_alphabetical_order(df_mapping)
    # Analysis.plot_time_to_start_cross_by_alphabetical_order(df_mapping)
    # Analysis.plot_speed_to_cross_by_average(df_mapping)
    # Analysis.plot_time_to_start_cross_by_average(df_mapping)
    # Analysis.correlation_matrix(df_mapping)
    
    # # Data to show on hover in scatter plots
    # hover_data = ["state", "country", "gdp_city_(billion_US)", "population_city", "population_country",
    #               "traffic_mortality", "literacy_rate", "person", "bicycle", "car", "motorcycle",
    #               "bus", "truck", "cellphone", "traffic_light", "stop_sign", "total_time", "speed_crossing",
    #               "speed_crossing_day", "speed_crossing_night", "time_crossing", "time_crossing_day", "time_crossing_night"]

    # # Speed of crossing vs time to start crossing
    # df = df_mapping[df_mapping["speed_crossing"] != 0]
    # df = df[df["time_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="speed_crossing",
    #                  y="time_crossing",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing speed (in m/s)',
    #                  yaxis_title='Pedestrian crossing decision time (in s)',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Speed of crossing during daytime vs time to start crossing during daytime
    # df = df_mapping[df_mapping["speed_crossing_day"] != 0]
    # df = df[df["time_crossing_day"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="speed_crossing_day",
    #                  y="time_crossing_day",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing speed during daytime (in m/s)',
    #                  yaxis_title='Pedestrian crossing decision time during daytime (in s)',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Speed of crossing during night time vs time to start crossing during night time
    # df = df_mapping[df_mapping["speed_crossing_night"] != 0]
    # df = df[df["time_crossing_night"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="speed_crossing_night",
    #                  y="time_crossing_night",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing speed during night time (in m/s)',
    #                  yaxis_title='Pedestrian crossing decision time during night time (in s)',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Speed of crossing vs population of city
    # df = df_mapping[df_mapping["speed_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="speed_crossing",
    #                  y="population_city",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing speed (in m/s)',
    #                  yaxis_title='Population of city',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Time to start crossing vs population of city
    # df = df_mapping[df_mapping["time_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="time_crossing",
    #                  y="population_city",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing decision time (in s)',
    #                  yaxis_title='Population of city',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Speed of crossing vs population of city
    # df = df_mapping[df_mapping["speed_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="speed_crossing",
    #                  y="population_city",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing speed (in m/s)',
    #                  yaxis_title='Population of city',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Time to start crossing vs population of city
    # df = df_mapping[df_mapping["time_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="time_crossing",
    #                  y="traffic_mortality",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing decision time (in s)',
    #                  yaxis_title='Traffic mortality',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Speed of crossing vs population of city
    # df = df_mapping[df_mapping["speed_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="speed_crossing",
    #                  y="traffic_mortality",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing speed (in m/s)',
    #                  yaxis_title='Traffic mortality',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Time to start crossing vs population of city
    # df = df_mapping[df_mapping["time_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="time_crossing",
    #                  y="literacy_rate",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing decision time (in s)',
    #                  yaxis_title='Literacy rate',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Speed of crossing vs population of city
    # df = df_mapping[df_mapping["speed_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="speed_crossing",
    #                  y="literacy_rate",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing speed (in m/s)',
    #                  yaxis_title='Literacy rate',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Time to start crossing vs population of city
    # df = df_mapping[df_mapping["time_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="time_crossing",
    #                  y="gini",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing decision time (in s)',
    #                  yaxis_title='Gini coefficient',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Speed of crossing vs population of city
    # df = df_mapping[df_mapping["speed_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="speed_crossing",
    #                  y="gini",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing speed (in m/s)',
    #                  yaxis_title='Gini coefficient',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Time to start crossing vs population of city
    # df = df_mapping[df_mapping["time_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="time_crossing",
    #                  y="traffic_index",
    #                  color="continent",
    #                  text="city",
    #                  # size="gdp_city_(billion_US)",
    #                  xaxis_title='Pedestrian crossing decision time (in s)',
    #                  yaxis_title='Traffic index',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
    # # Speed of crossing vs population of city
    # df = df_mapping[df_mapping["speed_crossing"] != 0]
    # df['state'] = df['state'].fillna('NA')
    # Analysis.scatter(df=df,
    #                  x="speed_crossing",
    #                  y="traffic_index",
    #                  color="continent",
    #                  text="city",
    #                  xaxis_title='Pedestrian crossing speed (in m/s)',
    #                  yaxis_title='Traffic index',
    #                  pretty_text=False,
    #                  save_file=True,
    #                  hover_data=hover_data,
    #                  hover_name="city",
    #                  legend_title="",
    #                  marginal_x=None,
    #                  marginal_y=None)
