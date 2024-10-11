import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import common
from custom_logger import CustomLogger
from logmod import logs
import statistics
import ast
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import pickle
from datetime import datetime

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# set template for plotly output
template = common.get_configs('plotly_template')

# File to store the city coordinates
pickle_file_coordinates = 'city_coordinates.pkl'
pickle_file_path = 'analysis_results.pkl'


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
    def save_plotly_figure(fig, filename, width=1600, height=900, scale=3):
        """Saves a Plotly figure as HTML, PNG, SVG, and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): Width of the PNG and EPS images in pixels. Defaults to 1600.
            height (int, optional): Height of the PNG and EPS images in pixels. Defaults to 900.
            scale (int, optional): Scaling factor for the PNG image. Defaults to 3.
        """
        # Create directory if it doesn't exist
        output_folder = "_output"
        os.makedirs(output_folder, exist_ok=True)

        # Save as HTML
        fig.write_html(os.path.join(output_folder, filename + ".html"))

        # Save as PNG
        fig.write_image(os.path.join(output_folder, filename + ".png"),
                        width=width, height=height, scale=scale)

        # Save as SVG
        fig.write_image(os.path.join(output_folder, filename + ".svg"),
                        format="svg")

        # Save as EPS
        fig.write_image(os.path.join(output_folder, filename + ".eps"),
                        width=width, height=height)

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
                if (abs(ann['x'] - other_ann['x']) < 0) and (abs(ann['y'] - other_ann['y']) < 0):
                    adjusted_ann['y'] += 0  # Adjust y-coordinate (can be modified as needed)

            # Append the adjusted annotation to the list
            adjusted_annotations.append(adjusted_ann)

        return adjusted_annotations

    @staticmethod
    def plot_scatter_diag(x, y, size, color, symbol, city, plot_name, x_label, y_label,
                          legend_x=0.887, legend_y=0.986, legend_font_size=24, tick_font_size=24,
                          label_font_size=24, need_annotations=False):
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
                            'South America': 'purple', 'Australia': 'brown'}

        # Define the mapping from "0" and "1" to "day" and "night"
        symbol_mapping = {"0": "Day", "1": "Night"}

        # Map the symbol list (which contains "0" and "1") to corresponding "day" and "night"
        mapped_symbol = [symbol_mapping[s] for s in symbol]

        # Create the scatter plot with hover_data for additional information (continents and sizes)
        fig = px.scatter(x=x, y=list(y.values()), size=size, color=color, symbol=symbol,
                         labels={"color": "Continent"}, color_discrete_map=continent_colors,
                         hover_data={"City": city, "Condition": mapped_symbol, "Continent": color,
                                     "GMP per capita": size})

        # Customize the hovertemplate to only show the fields you want
        fig.update_traces(
            hovertemplate="<br>City=%{customdata[0]}<br>Condition=%{customdata[1]}<br>GMP per capita=%{customdata[2]}<extra></extra>"  # noqa:E501
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
        symbols_legend = {'diamond': 'Night', 'circle': 'Day'}
        for symbol, description in symbols_legend.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                     marker=dict(symbol=symbol, color='rgba(0,0,0,0)',
                                                 line=dict(color='black', width=2)), name=description))

        # Adding annotations for locations
        annotations = []
        if need_annotations:
            for i, key in enumerate(y.keys()):
                annotations.append(
                    dict(x=x[i], y=list(y.values())[i], text=city[i], showarrow=False)
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

        # Show the plot
        fig.show()

        # Save the plot
        Analysis.save_plotly_figure(fig, plot_name)

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
            country = row["country"]
            gdp = row["gdp_city_(billion_US)"]
            population = row["population_city"]
            population_country = row["population_country"]
            traffic_mortality = row["traffic_mortality"]
            continent = row["continent"]
            literacy_rate = row["literacy_rate"]
            avg_height = row["avg_height"]
            iso_country = row["ISO_country"]

            # Iterate through each video, start time, end time, and time of day
            for video, start, end, time_of_day_ in zip(video_ids, start_times, end_times, time_of_day):
                # Check if the current video matches the specified ID
                if video == id:
                    counter = 0
                    # Iterate through each start time
                    for s in start:
                        # Check if the start time matches the specified start time
                        if int(start_) == s:
                            # Return relevant information once found
                            return (video, s, end[counter], time_of_day_[counter], city,
                                    country, (gdp/population), population, population_country,
                                    traffic_mortality, continent, literacy_rate, avg_height, iso_country)
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
    def get_value(df, column_name, column_value, target_column):
        """
        Retrieves a value from the target_column based on the condition
        that the column_name matches the column_value.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the mapping file.
        column_name (str): The column to search for the matching value.
        column_value (str): The value to search for in column_name.
        target_column (str): The column from which to retrieve the corresponding value.

        Returns:
        Any: The value from target_column that corresponds to the matching column_value in column_name.
        """
        # Filter the DataFrame where column_name has the value column_value
        result = df[df[column_name] == column_value][target_column]

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
        cities = df_mapping["city"]
        countries = df_mapping["country"]

        # Create the country list to highlight in the choropleth map
        countries_set = set(countries)  # Use set to avoid duplicates
        if "Denmark" in countries_set:
            countries_set.add('Greenland')

        # Create a DataFrame for highlighted countries with a value (same for all to have the same color)
        df = pd.DataFrame({'country': list(countries_set), 'value': 1})

        # Create a choropleth map using Plotly with grey color for countries
        fig = px.choropleth(df, locations="country", locationmode="country names",
                            color="value", hover_name="country", hover_data={'value': False, 'country': False},
                            color_continuous_scale=["#808080", "#808080"], labels={'value': 'Highlighted'})

        # Update layout to remove Antarctica, Easter Island, remove the color bar, and set ocean color
        fig.update_layout(
            coloraxis_showscale=False,  # Remove color bar
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular',
                showlakes=False,
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
        for i, city in enumerate(cities):
            city_country = f"{city}, {countries[i]}"  # Combine city and country
            lat, lon = Analysis.get_coordinates(city_country, city_coordinates)  # type: ignore
            if lat and lon:
                city_coords.append({'city': city, 'lat': lat, 'lon': lon})

        # Save the updated city coordinates back to the pickle file
        with open(pickle_file_coordinates, 'wb') as f:
            pickle.dump(city_coordinates, f)

        if city_coords:
            city_df = pd.DataFrame(city_coords)
            city_trace = px.scatter_geo(city_df, lat='lat', lon='lon',
                                        hover_name='city',  # Display city name on hover
                                        hover_data={'lat': False, 'lon': False}  # Only show city name
                                        )
            # Update the city markers to be red and adjust size
            city_trace.update_traces(marker=dict(color="red", size=10))

            # Add the scatter_geo trace to the choropleth map
            fig.add_trace(city_trace.data[0])

        fig.show()

        # Save and display the figure
        Analysis.save_plotly_figure(fig, "world_map")

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
    def time_to_cross(dataframe, ids):
        """Calculates the time taken for each object with specified IDs to cross the road.

        Args:
            dataframe (DataFrame): The DataFrame (csv file) containing object data.
            ids (list): A list of unique IDs of objects which are crossing the road.

        Returns:
            dict: A dictionary where keys are object IDs and values are the time taken for
            each object to cross the road, in seconds.
        """

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
                            var[id] = count/30
                            break

            else:
                for value in sorted_grp['X-center']:
                    if value == x_max:
                        flag = 1
                    if flag == 1:
                        count += 1
                        if value == x_min:
                            # Calculate time taken for crossing and store in dictionary
                            var[id] = count / 30
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
                (video, start, end, time_of_day, city, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso_country) = result

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
                if f"{city}_{condition}" in info:
                    previous_value = info[f"{city}_{condition}"]
                    # Extracting the old number of detected mobiles
                    previous_value = previous_value * no_person[f"{city}_{condition}"] * total_time[
                        f"{city}_{condition}"] / 1000 / 60

                    # Summing up the previous value and the new value
                    total_value = previous_value + mobile_ids
                    no_person[f"{city}_{condition}"] += num_person
                    total_time[f"{city}_{condition}"] += duration

                    # Normalising with respect to total person detected and time
                    info[f"{city}_{condition}"] = (((total_value * 60) / total_time[f"{city}_{condition}"]
                                                    ) / no_person[f"{city}_{condition}"]) * 1000
                    continue  # Skip saving the variable in plotting variables
                else:
                    no_person[f"{city}_{condition}"] = num_person
                    total_time[f"{city}_{condition}"] = duration

                    """Normalising the detection with respect to time and numvber of person in the video.
                    Multiplied by 1000 to increase the value to look better in plotting."""

                    avg_cell_phone = (((mobile_ids * 60) / time_[-1]) / num_person) * 1000
                    info[f"{city}_{condition}"] = avg_cell_phone

            else:
                # Handle the case where no data was found for the given key
                logger.error(f"No matching data found for key: {key}")

        return info

    @staticmethod
    def calculate_vehicle_detected(df_mapping, dfs, data, motorcycle=1, car=1, bus=1, truck=1):
        """Plots the relationship between vehicle detection and crossing time.

        Args:
            df_mapping (DataFrame): DataFrame containing mapping information.
            dfs (dict): Dictionary of DataFrames containing video data.
            data (dict): Dictionary containing information about which object is crossing.
            motorcycle (int, optional): Flag to include motorcycles. Default is 1.
            car (int, optional): Flag to include cars. Default is 1.
            bus (int, optional): Flag to include buses. Default is 1.
            truck (int, optional): Flag to include trucks. Default is 1.
        """

        info = {}
        time_ = []

        # Iterate through each video DataFrame
        for key, value in dfs.items():
            result = Analysis.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                # Unpack the result since it's not None
                (video, start, end, time_of_day, city, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso_country) = result

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

                else:
                    logger.info("No plot generated")

                vehicle_ids = vehicle_ids["Unique Id"].unique()

                if vehicle_ids is None:
                    continue

                # Calculate normalized vehicle detection rate
                new_value = ((len(vehicle_ids)/time_[-1]) * 60)

                # Update the information dictionary
                if f"{city}_{condition}" in info:
                    previous_value = info[f"{city}_{condition}"]
                    info[f"{city}_{condition}"] = (previous_value + new_value) / 2
                    continue
                else:
                    info[f"{city}_{condition}"] = new_value

        return info

    @staticmethod
    def calculate_speed_of_crossing(df_mapping, dfs, data, person_id=0):
        avg_speed, no_people = {}, {},
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
                (_, start, end, condition, city, country, gdp_, population, population_country, traffic_mortality_,
                 continent, literacy_rate, avg_height, iso_country) = result
                # Convert country to uppercase
                # city = city.upper()

                value = dfs.get(key)

                # Store the country associated with each city
                city_country_map_[city] = iso_country

                # Calculate the duration of the video
                duration = end - start
                time_.append(duration)

                grouped = value.groupby('Unique Id')
                speed = []
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

                    speed.append(speed_)
                if len(speed) == 0:
                    continue
                no_people[f'{city}_{condition}'] = len(speed)

                if f'{city}_{condition}' in avg_speed:
                    old_count = avg_speed[f'{city}_{condition}']
                    new_count = old_count * no_people[f'{city}_{condition}'] + sum(speed)
                    avg_speed[f'{city}_{condition}'] = new_count / (no_people[f'{city}_{condition}'] + len(speed))
                else:
                    avg_speed[f'{city}_{condition}'] = sum(speed) / len(speed)

        return avg_speed

    @staticmethod
    def time_to_start_cross(df_mapping, dfs, data, person_id=0):
        time_dict, no_people_ = {}, {}
        for key, df in dfs.items():
            data_cross = {}
            crossed_ids = df[(df["YOLO_id"] == person_id)]

            # Extract relevant information using the find_values function
            result = Analysis.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                (_, start, end, condition, city, country, gdp_, population, population_country, traffic_mortality_,
                 continent, literacy_rate, avg_height, iso_country) = result

                # Convert country to uppercase
                # city = city.upper()

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

                if len(data_cross) == 0:  # Skip if no crossing events detected
                    continue

                if f'{city}_{condition}' in time_dict:
                    old_count = time_dict[f'{city}_{condition}']
                    new_count = old_count * no_people_[f'{city}_{condition}'] + (sum(data_cross.values()) / 30)
                    no_people_[f'{city}_{condition}'] += len(data_cross)
                    time_dict[f'{city}_{condition}'] = new_count / no_people_[f'{city}_{condition}']
                else:
                    time_dict[f'{city}_{condition}'] = ((sum(data_cross.values()) / 30) / len(data_cross))
                    no_people_[f'{city}_{condition}'] = len(data_cross)

        return time_dict

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
                (_, start, end, time_of_day, city, country, gdp_, population, population_country, traffic_mortality_,
                 continent, literacy_rate, avg_height, iso_country) = result

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
                if f'{city}_{condition}' in info:
                    old_count = info[f'{city}_{condition}']
                    new_count = (old_count * duration_.get(f'{city}_{condition}', 0)) + count_
                    if f'{city}_{condition}' in duration_:
                        duration_[f'{city}_{condition}'] = duration_.get(f'{city}_{condition}', 0) + count
                    else:
                        duration_[f'{city}_{condition}'] = count
                    info[f'{city}_{condition}'] = new_count / duration_.get(f'{city}_{condition}', 0)
                    continue
                else:
                    info[f'{city}_{condition}'] = count_

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

        # For a specific id of a person search for the first and last occurrence of that id and see if the traffic
        # light was present between it or not. Only getting those unique_id of the person who crosses the road.

        # Loop through each video data
        for key, df in data.items():
            counter_1, counter_2 = {}, {}
            counter_exists, counter_nt_exists = 0, 0

            # Extract relevant information using the find_values function
            result = Analysis.find_values_with_video_id(df_mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:

                (_, start, end, time_of_day, city, country, gdp_, population, population_country, traffic_mortality_,
                 continent, literacy_rate, avg_height, iso_country) = result

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

                counter_1[f'{city}_{condition}'] = counter_1.get(f'{city}_{condition}', 0) + var_exist[key]
                counter_2[f'{city}_{condition}'] = counter_2.get(f'{city}_{condition}', 0) + var_nt_exist[key]

                if (counter_1[f'{city}_{condition}'] + counter_2[f'{city}_{condition}']) == 0:
                    # Gives an error of division by 0
                    continue
                else:
                    if f'{city}_{condition}' in ratio:
                        ratio[f'{city}_{condition}'] = ((counter_2[f'{city}_{condition}'] * 100) /
                                                        (counter_1[f'{city}_{condition}'] +
                                                        counter_2[f'{city}_{condition}']))
                        continue
                    # If already present, the array below will be filled multiple times
                    else:
                        ratio[f'{city}_{condition}'] = ((counter_2[f'{city}_{condition}'] * 100) /
                                                        (counter_1[f'{city}_{condition}'] +
                                                        counter_2[f'{city}_{condition}']))
        return ratio

# Plotting functions:

    @staticmethod
    def speed_and_time_to_start_cross(df_mapping, dfs, data):
        final_dict = {}
        speed_values = Analysis.calculate_speed_of_crossing(df_mapping, dfs, data)
        time_values = Analysis.time_to_start_cross(df_mapping, dfs, data)

        # Check if both 'speed' and 'time' are valid dictionaries
        if speed_values is None or time_values is None:
            raise ValueError("Either 'speed' or 'time' returned None, please check the input data or calculations.")

        # Remove the ones where there is data missing for a specific country and condition
        common_keys = speed_values.keys() & time_values.keys()

        # Retain only the key-value pairs where the key is present in both dictionaries
        speed_values = {key: speed_values[key] for key in common_keys}
        time_values = {key: time_values[key] for key in common_keys}

        # Now populate the final_dict with city-wise data
        for city_condition, speed in speed_values.items():
            city, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = Analysis.get_value(df_mapping, "city", city, "country")
            iso_code = Analysis.get_value(df_mapping, "city", city, "ISO_country")
            if country or iso_code is not None:

                # Initialize the city's dictionary if not already present
                if city not in final_dict:
                    final_dict[city] = {
                        "speed_0": None, "speed_1": None, "time_0": None, "time_1": None,
                        "country": country, "iso": iso_code
                    }
                # Populate the corresponding speed and time based on the condition
                final_dict[city][f"speed_{condition}"] = speed
                if f'{city}_{condition}' in time_values:
                    final_dict[city][f"time_{condition}"] = time_values[f'{city}_{condition}']

        # Extract all valid speed_0 and speed_1 values along with their corresponding cities
        diff_speed_values = [(city, abs(data['speed_0'] - data['speed_1']))
                             for city, data in final_dict.items()
                             if data['speed_0'] is not None and data['speed_1'] is not None]

        if diff_speed_values:
            # Sort the list by the absolute difference and get the top 5 and bottom 5
            sorted_diff_speed_values = sorted(diff_speed_values, key=lambda x: x[1], reverse=True)

            top_5_max_speed = sorted_diff_speed_values[:5]  # Top 5 maximum differences
            top_5_min_speed = sorted_diff_speed_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("\nTop 5 cities with max |speed_0 - speed_1| differences:")
            for city, diff in top_5_max_speed:
                logger.info(f"{city}: {diff}")

            logger.info("\nTop 5 cities with min |speed_0 - speed_1| differences:")
            for city, diff in top_5_min_speed:
                logger.info(f"{city}: {diff}")
        else:
            logger.info("\nNo valid speed_0 and speed_1 values found for comparison.")

        # Extract all valid time_0 and time_1 values along with their corresponding cities
        diff_time_values = [(city, abs(data['time_0'] - data['time_1']))
                            for city, data in final_dict.items()
                            if data['time_0'] is not None and data['time_1'] is not None]

        if diff_time_values:
            sorted_diff_time_values = sorted(diff_time_values, key=lambda x: x[1], reverse=True)

            top_5_max = sorted_diff_time_values[:5]  # Top 5 maximum differences
            top_5_min = sorted_diff_time_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("\nTop 5 cities with max |time_0 - time_1| differences:")
            for city, diff in top_5_max:
                logger.info(f"{city}: {diff}")

            logger.info("\nTop 5 cities with min |time_0 - time_1| differences:")
            for city, diff in top_5_min:
                logger.info(f"{city}: {diff}")
        else:
            logger.info("\nNo valid time_0 and time_1 values found for comparison.")

        # Filtering out entries where speed_0 or speed_1 is None
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

            logger.info(f"\nCity with max speed_0: {max_speed_city_0} with speed_0 = {max_speed_value_0}")
            logger.info(f"\nCity with min speed_0: {min_speed_city_0} with speed_0 = {min_speed_value_0}")

        if filtered_dict_s_1:
            max_speed_city_1 = max(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            min_speed_city_1 = min(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            max_speed_value_1 = filtered_dict_s_1[max_speed_city_1]["speed_1"]
            min_speed_value_1 = filtered_dict_s_1[min_speed_city_1]["speed_1"]

            logger.info(f"\nCity with max speed at night: {max_speed_city_1} with speed = {max_speed_value_1}")
            logger.info(f"City with min speed at night: {min_speed_city_1} with speed = {min_speed_value_1}")

        # Find city with max and min time_0 and time_1
        if filtered_dict_t_0:
            max_time_city_0 = max(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            min_time_city_0 = min(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            max_time_value_0 = filtered_dict_t_0[max_time_city_0]["time_0"]
            min_time_value_0 = filtered_dict_t_0[min_time_city_0]["time_0"]

            logger.info(f"\nCity with max time_0: {max_time_city_0} with time_0 = {max_time_value_0}")
            logger.info(f"City with min time_0: {min_time_city_0} with time_0 = {min_time_value_0}")

        if filtered_dict_t_1:
            max_time_city_1 = max(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            min_time_city_1 = min(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            max_time_value_1 = filtered_dict_t_1[max_time_city_1]["time_1"]
            min_time_value_1 = filtered_dict_t_1[min_time_city_1]["time_1"]

            logger.info(f"\nCity with max time_1: {max_time_city_1} with time_1 = {max_time_value_1}")
            logger.info(f"City with min time_1: {min_time_city_1} with time_1 = {min_time_value_1}")

        # Extract valid speed and time values and calculate statistics
        speed_0_values = [data['speed_0'] for data in final_dict.values() if data['speed_0'] is not None]
        speed_1_values = [data['speed_1'] for data in final_dict.values() if data['speed_1'] is not None]
        time_0_values = [data['time_0'] for data in final_dict.values() if data['time_0'] is not None]
        time_1_values = [data['time_1'] for data in final_dict.values() if data['time_1'] is not None]

        if speed_0_values:
            mean_speed_0 = statistics.mean(speed_0_values)
            sd_speed_0 = statistics.stdev(speed_0_values) if len(speed_0_values) > 1 else 0
            logger.info(f"\nMean of speed during day time: {mean_speed_0}")
            logger.info(f"Standard deviation of speed during day time: {sd_speed_0}")
        else:
            logger.error("No valid speed during day time values found.")

        if speed_1_values:
            mean_speed_1 = statistics.mean(speed_1_values)
            sd_speed_1 = statistics.stdev(speed_1_values) if len(speed_1_values) > 1 else 0
            logger.info(f"\nMean of speed during night time: {mean_speed_1}")
            logger.info(f"Standard deviation of speed during night time: {sd_speed_1}")
        else:
            logger.error("No valid speed during night time values found.")

        if time_0_values:
            mean_time_0 = statistics.mean(time_0_values)
            sd_time_0 = statistics.stdev(time_0_values) if len(time_0_values) > 1 else 0
            logger.info(f"\nMean of time during day time: {mean_time_0}")
            logger.info(f"Standard deviation of time during day time: {sd_time_0}")
        else:
            logger.error("No valid time during day time values found.")

        if time_1_values:
            mean_time_1 = statistics.mean(time_1_values)
            sd_time_1 = statistics.stdev(time_1_values) if len(time_1_values) > 1 else 0
            logger.info(f"\nMean of time during night time: {mean_time_1}")
            logger.info(f"Standard deviation of time during night time: {sd_time_1}")
        else:
            logger.error("No valid time during night time values found.")

        # Extract city, condition, and count_ from the info dictionary
        cities, conditions_, counts = [], [], []
        for key, value in time_values.items():
            city, condition = key.split('_')
            cities.append(city)
            conditions_.append(condition)
            counts.append(value)

        # Combine keys from speed and time to ensure we include all available cities and conditions
        all_keys = set(speed_values.keys()).union(set(time_values.keys()))

        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in all_keys]))

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
        day_time_dict = [final_dict[city]['time_0'] for city in cities_ordered]
        night_time_dict = [final_dict[city]['time_1'] for city in cities_ordered]

        # Ensure that plotting uses cities_ordered
        assert len(cities_ordered) == len(day_avg_speed) == len(night_avg_speed) == len(
            day_time_dict) == len(night_time_dict), "Lengths of lists don't match!"

        # Make sure the lengths match
        assert len(cities) == len(day_avg_speed) == len(night_avg_speed) == len(
            day_time_dict) == len(night_time_dict), "Lengths of lists don't match!"

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups

        fig = make_subplots(
            rows=num_cities_per_col * 2, cols=2,  # Two columns
            vertical_spacing=0,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col * 2),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            # Row for speed (Day and Night)
            row = 2 * i + 1
            if day_avg_speed[i] is not None and night_avg_speed[i] is not None:
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[city], orientation='h',
                    name=f"{city} speed during day", marker=dict(color='#FFA15A'), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[city], orientation='h',
                    name=f"{city} speed during night", marker=dict(color='#19D3F3'), text=[''],
                    textposition='auto', showlegend=False), row=row, col=1)

            elif day_avg_speed[i] is not None:  # Only day data available
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[city], orientation='h',
                    name=f"{city} speed during day", marker=dict(color='#FFA15A'), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_avg_speed[i] is not None:  # Only night data available
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[city], orientation='h',
                    name=f"{city} speed during night", marker=dict(color='#19D3F3'), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            # Row for time (Day and Night)
            row = 2 * i + 2
            if day_time_dict[i] is not None and night_time_dict[i] is not None:
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[city], orientation='h',
                    name=f"{city} time during day", marker=dict(color='#FF6692'),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[city], orientation='h',
                    name=f"{city} time during night", marker=dict(color='#B6E880'), text=[''],
                    textposition='auto', showlegend=False), row=row, col=1)

            elif day_time_dict[i] is not None:  # Only day time data available
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[city], orientation='h',
                    name=f"{city} time during day", marker=dict(color='#FF6692'),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_time_dict[i] is not None:  # Only night time data available
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[city], orientation='h',
                    name=f"{city} time during night", marker=dict(color='#B6E880'),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

        # Similarly for the right column
        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            row = 2 * i + 1
            idx = num_cities_per_col + i
            if day_avg_speed[idx] is not None and night_avg_speed[idx] is not None:
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[city], orientation='h',
                    name=f"{city} speed during day", marker=dict(color='#FFA15A'), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[city], orientation='h',
                    name=f"{city} speed during night", marker=dict(color='#19D3F3'), text=[''],
                    textposition='inside', showlegend=False), row=row, col=2)

            elif day_avg_speed[idx] is not None:
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[city], orientation='h',
                    name=f"{city} speed during day", marker=dict(color='#FFA15A'), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_avg_speed[idx] is not None:
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[city], orientation='h',
                    name=f"{city} speed during night", marker=dict(color='#19D3F3'), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            row = 2 * i + 2
            if day_time_dict[idx] is not None and night_time_dict[idx] is not None:
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[city], orientation='h',
                    name=f"{city} time during day", marker=dict(color='#FF6692'),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[city], orientation='h',
                    name=f"{city} time during night", marker=dict(color='#B6E880'), text=[''],
                    textposition='inside', showlegend=False), row=row, col=2)

            elif day_time_dict[idx] is not None:  # Only day time data available
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[city], orientation='h',
                    name=f"{city} time during day", marker=dict(color='#FF6692'),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_time_dict[idx] is not None:  # Only night time data available
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[city], orientation='h',
                    name=f"{city} time during night", marker=dict(color='#B6E880'),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_avg_speed[i] if day_avg_speed[i] is not None else 0) +
            (night_avg_speed[i] if night_avg_speed[i] is not None else 0)
            for i in range(len(cities))
        ]) if cities else 0

        # Use below to have two different scale in x axis for speed and time
        # max_value_time = max([
        #     (day_time_dict[i] if day_time_dict[i] is not None else 0) +
        #     (night_time_dict[i] if night_time_dict[i] is not None else 0)
        #     for i in range(len(cities))
        # ]) if cities else 0

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
        fig.update_xaxes(title_text="Time taken to start crossing the road (s)", titlefont=dict(size=40),
                         tickfont=dict(size=40), ticks='outside', ticklen=10, tickwidth=2, tickcolor='black',
                         row=num_cities_per_col * 2, col=1)
        fig.update_xaxes(title_text="Time taken to start crossing the road (s)", titlefont=dict(size=40),
                         tickfont=dict(size=40), ticks='outside', ticklen=10, tickwidth=2, tickcolor='black',
                         row=num_cities_per_col * 2, col=2)

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

        # Function to add vertical legend annotations
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

        # Define the legend items
        legend_items = [
            {"name": "Speed during day", "color": "#FFA15A"},
            {"name": "Speed during night", "color": "#19D3F3"},
            {"name": "Time during day", "color": "#FF6692"},
            {"name": "Time during night", "color": "#B6E880"},
        ]

        # Add vertical legends with the positions you will provide
        x_legend_position = 0.30  # Position close to the left edge
        y_legend_start_bottom = 0.65  # Lower position to the bottom left corner

        # Add the vertical legends at the top and bottom
        add_vertical_legend_annotations(fig, legend_items, x_position=x_legend_position,
                                        y_start=y_legend_start_bottom, spacing=0.02, font_size=40)

        # Add a box around the legend
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=x_legend_position,  # Adjust x0 to control the left edge of the box
            y0=y_legend_start_bottom + 0.02,  # Adjust y0 to control the top of the box
            x1=x_legend_position + 0.195,  # Adjust x1 to control the right edge of the box
            y1=y_legend_start_bottom - len(legend_items) * 0.03 + 0.04,  # Adjust y1 to control the bottom of the box
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
        x_position_left = -0.01  # Position for the left column
        x_position_right = 1.02  # Position for the right column
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
            tickfont=dict(size=15, family="Arial", color="black", weight="bold"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=150, b=180))
        fig.show()
        Analysis.save_plotly_figure(fig, "consolidate",  width=2480, height=3000, scale=1)

    @staticmethod
    def time_to_start_crossing_vs_literacy(df_mapping, dfs, data, need_annotations=False):
        literacy, continents, gdp = [], [], []  # Lists for literacy, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        time = Analysis.time_to_start_cross(df_mapping, dfs, data)

        for key, value in time.items():
            city, condition = key.split('_')
            if need_annotations:
                cities.append(city)
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            literacy.append(float(Analysis.get_value(df_mapping, "city", city, "literacy_rate")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "continent"))
            gdp.append(float(Analysis.get_value(df_mapping, "city", city,
                                                "gdp_city_(billion_US)"))/float(Analysis.get_value(  # type: ignore
                                                    df_mapping, "city", city, "population_city")))  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=literacy, y=time, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="time_to_start_crossing_vs_literacy",
                                   x_label="Literacy rate in the country (in percentage)",
                                   y_label="Time taken by pedestrian to start crossing the road (in s)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def time_to_start_crossing_vs_traffic_mortality(df_mapping, dfs, data, need_annotations=False):
        traffic_deaths, continents, gdp = [], [], []  # Lists for traffic related deaths, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        time = Analysis.time_to_start_cross(df_mapping, dfs, data)

        for key, value in time.items():
            city, condition = key.split('_')
            if need_annotations:
                cities.append(city)
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            traffic_deaths.append(float(Analysis.get_value(df_mapping, "city", city, "literacy_rate")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "continent"))
            gdp.append(float(Analysis.get_value(df_mapping, "city", city,
                                                "gdp_city_(billion_US)"))/float(Analysis.get_value(  # type: ignore
                                                    df_mapping, "city", city, "population_city")))  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=traffic_deaths, y=time, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="traffic_safety_vs_literacy",
                                   x_label="Literacy rate in the country (in percentage)",
                                   y_label="Number of traffic instruments detected (normalised)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def traffic_safety_vs_literacy(df_mapping, dfs, need_annotations=False):
        """Plots traffic safety vs literacy.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
        """
        literacy, continents, gdp = [], [], []  # Lists for literacy, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        info = Analysis.traffic_signs(df_mapping, dfs)

        for key, value in info.items():
            city, condition = key.split('_')
            if need_annotations:
                cities.append(city)
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            literacy.append(float(Analysis.get_value(
                df_mapping, "city", city, "traffic_mortality")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "continent"))
            gdp.append(float(Analysis.get_value(df_mapping, "city", city,
                                                "gdp_city_(billion_US)"))/float(Analysis.get_value(  # type: ignore
                                                    df_mapping, "city", city, "population_city")))  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=literacy, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="time_to_start_crossing_vs_literacy",
                                   x_label="Traffic mortality rate (per 100,000 population)",
                                   y_label="Time taken by pedestrian to start crossing the road (in seconds)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def plot_cell_phone_vs_traffic_mortality(df_mapping, dfs, need_annotations=False):
        """Plots the relationship between average cell phone usage per person detected vs. traffic mortality.

        Args:
            df_mapping (DataFrame): DataFrame containing mapping information.
            dfs (dict): Dictionary of DataFrames containing video data.
        """
        traffic_deaths, continents, gdp = [], [], []  # Lists for traffic related deaths, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        info = Analysis.calculate_cell_phones(df_mapping, dfs)
        for key, value in info.items():
            city, condition = key.split('_')
            if need_annotations:
                cities.append(city)
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            traffic_deaths.append(float(Analysis.get_value(
                df_mapping, "city", city, "traffic_mortality")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "continent"))
            gdp.append(float(Analysis.get_value(df_mapping, "city", city,
                                                "gdp_city_(billion_US)"))/float(Analysis.get_value(  # type: ignore
                                                    df_mapping, "city", city, "population_city")))  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=traffic_deaths, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="cell_phone_vs_traffic_mortality",
                                   x_label="Traffic mortality rate (per 100,000 population)",
                                   y_label="Number of Mobile detected in the video (normalised)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def vehicle_vs_cross_time(df_mapping, dfs, data, need_annotations=False):
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
        info = Analysis.calculate_vehicle_detected(df_mapping, dfs, data, motorcycle=1, car=1, bus=1, truck=1)
        time = Analysis.time_to_start_cross(df_mapping, dfs, data)

        for key, value in info.items():
            city, condition = key.split('_')
            if need_annotations:
                cities.append(city)
            else:
                cities.append("")
            conditions.append(condition)
            time_cal.append(time.get(key))
            counts.append(value)
            continents.append(Analysis.get_value(df_mapping, "city", city, "continent"))
            gdp.append(float(Analysis.get_value(df_mapping, "city", city,
                                                "gdp_city_(billion_US)"))/float(Analysis.get_value(  # type: ignore
                                                    df_mapping, "city", city, "population_city")))  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=time_cal, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="cell_phone_vs_traffic_mortality",
                                   x_label="Traffic mortality rate (per 100,000 population)",
                                   y_label="Number of Mobile detected in the video (normalised)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping, dfs, data, need_annotations=False):
        """Plots traffic mortality rate vs percentage of crossing events without traffic light.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
            data (dict): Dictionary containing pedestrian crossing data.
        """
        traffic_deaths, continents, gdp = [], [], []  # Lists for traffic related deaths, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        info = Analysis.crossing_event_wt_traffic_light(df_mapping, dfs, data)

        for key, value in info.items():
            city, condition = key.split('_')
            if need_annotations:
                cities.append(city)
            else:
                cities.append("")
            conditions.append(condition)
            counts.append(value)
            traffic_deaths.append(float(Analysis.get_value(df_mapping, "city", city, "literacy_rate")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "continent"))
            gdp.append(float(Analysis.get_value(df_mapping, "city", city,
                                                "gdp_city_(billion_US)"))/float(Analysis.get_value(  # type: ignore
                                                    df_mapping, "city", city, "population_city")))  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=traffic_deaths, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="traffic_mortality_vs_crossing_event_wt_traffic_light",
                                   x_label="Literacy rate in the country (in percentage)",
                                   y_label="Percentage of Crossing Event without traffic light (normalised)",
                                   legend_x=0.07, legend_y=0.96)

    @staticmethod
    def plot_traffic_safety_vs_traffic_mortality(df_mapping, dfs, need_annotations=False):
        """Plots traffic safety vs traffic mortality.

        Args:
            df_mapping (dict): Mapping of video keys to relevant information.
            dfs (dict): Dictionary of DataFrames containing pedestrian data.
        """
        traffic_deaths, continents, gdp = [], [], []  # Lists for literacy, continents, and GDP
        conditions = []  # Lists for conditions, time, and city
        cities, counts = [], []
        info = Analysis.traffic_signs(df_mapping, dfs)

        for key, value in info.items():
            city, condition = key.split('_')
            cities.append(city)
            conditions.append(condition)
            counts.append(value)
            traffic_deaths.append(float(Analysis.get_value(
                df_mapping, "city", city, "traffic_mortality")))  # type: ignore
            continents.append(Analysis.get_value(df_mapping, "city", city, "continent"))
            gdp.append(float(Analysis.get_value(df_mapping, "city", city,
                                                "gdp_city_(billion_US)"))/float(Analysis.get_value(  # type: ignore
                                                    df_mapping, "city", city, "population_city")))  # type: ignore

        # Plot the scatter diagram
        Analysis.plot_scatter_diag(x=traffic_deaths, y=info, size=gdp, color=continents, symbol=conditions,
                                   city=cities, plot_name="time_to_start_crossing_vs_literacy",
                                   x_label="Traffic mortality rate (per 100,000 population)",
                                   y_label="Number of traffic instruments detected (normalised)",
                                   need_annotations=need_annotations, legend_x=0.887, legend_y=0.96)


# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")

    # Stores the mapping file
    df_mapping = pd.read_csv("mapping.csv")

    # Stores the content of the csv file in form of {name_time: content}
    dfs = Analysis.read_csv_files(common.get_configs('data'))

    pedestrian_crossing_count, data = {}, {}
    person_counter, bicycle_counter, car_counter, motorcycle_counter = 0, 0, 0, 0
    bus_counter, truck_counter, cellphone_counter, traffic_light_counter, stop_sign_counter = 0, 0, 0, 0, 0

    logger.info("Duration of videos in seconds: {}", Analysis.calculate_total_seconds(df_mapping))
    logger.info("Total number of videos: {}", Analysis.calculate_total_videos(df_mapping))
    country, number = Analysis.get_unique_values(df_mapping, "country")
    logger.info("Total number of countries: {}", number)
    Analysis.get_world_plot(df_mapping)

    if os.path.exists(pickle_file_path):
        # Load the data from the pickle file
        with open(pickle_file_path, 'rb') as file:
            (data, pedestrian_crossing_count, person_counter, bicycle_counter, car_counter,
             motorcycle_counter, bus_counter, truck_counter, cellphone_counter,
             traffic_light_counter, stop_sign_counter) = pickle.load(file)
        logger.info("Loaded analysis results from pickle file.")
    else:
        # Loop over rows of data
        for key, value in dfs.items():
            logger.info("Analysing data from {}.", key)

            # Get the number of number and unique id of the object crossing the road
            count, ids = Analysis.pedestrian_crossing(dfs[key], 0.45, 0.55, 0)

            # Saving it in a dictionary in: {name_time: count, ids}
            pedestrian_crossing_count[key] = {"count": count, "ids": ids}

            # Saves the time to cross in form {name_time: {id(s): time(s)}}
            data[key] = Analysis.time_to_cross(dfs[key], pedestrian_crossing_count[key]["ids"])

            # Calculate the total number of different objects detected
            person_counter += Analysis.count_object(dfs[key], 0)
            bicycle_counter += Analysis.count_object(dfs[key], 1)
            car_counter += Analysis.count_object(dfs[key], 2)
            motorcycle_counter += Analysis.count_object(dfs[key], 3)
            bus_counter += Analysis.count_object(dfs[key], 5)
            truck_counter += Analysis.count_object(dfs[key], 7)
            cellphone_counter += Analysis.count_object(dfs[key], 67)
            traffic_light_counter += Analysis.count_object(dfs[key], 9)
            stop_sign_counter += Analysis.count_object(dfs[key], 11)

        # Save the results to a pickle file
        with open(pickle_file_path, 'wb') as file:
            pickle.dump((data, pedestrian_crossing_count, person_counter, bicycle_counter, car_counter,
                         motorcycle_counter, bus_counter, truck_counter, cellphone_counter,
                         traffic_light_counter, stop_sign_counter), file)
        logger.info("Analysis results saved to pickle file.")

    logger.info(f"person: {person_counter} ; bicycle: {bicycle_counter} ; car: {car_counter}")
    logger.info(f"motorcycle: {motorcycle_counter} ; bus: {bus_counter} ; truck: {truck_counter}")
    logger.info(f"cellphone: {cellphone_counter}; traffic light: {traffic_light_counter}; sign: {stop_sign_counter}")

    Analysis.speed_and_time_to_start_cross(df_mapping, dfs, data)
    Analysis.time_to_start_crossing_vs_literacy(df_mapping, dfs, data)
    Analysis.time_to_start_crossing_vs_traffic_mortality(df_mapping, dfs, data)
    Analysis.traffic_safety_vs_literacy(df_mapping, dfs)
    Analysis.plot_cell_phone_vs_traffic_mortality(df_mapping, dfs)
    Analysis.vehicle_vs_cross_time(df_mapping, dfs, data)
    Analysis.traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping, dfs, data)
    Analysis.plot_traffic_safety_vs_traffic_mortality(df_mapping, dfs)

    logger.info("Analysis completed.")
