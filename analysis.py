"""Summary

Attributes:
    logger (TYPE): Description
    template (TYPE): Description
"""
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from statistics import mean
import common
from custom_logger import CustomLogger
from logmod import logs
import statistics
import ast

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# set template for plotly output
template = common.get_configs('plotly_template')

# List of things that YOLO can detect:
# YOLO_id = {
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
#     9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
#     16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
#     24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',30: 'skis',
#     31: 'snowboard',32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',36: 'skateboard',
#     37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',43: 'knife',
#     44: 'spoon',45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
#     51: 'carrot', 52: 'hot dog',53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',57: 'couch',58: 'potted plant',
#     59: 'bed',60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',65: 'remote', 66: 'keyboard',
#     67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
#     74: 'clock',75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
# }


# Read the csv files and stores them as a dictionary in form {Unique_id : CSV}
def read_csv_files(folder_path):
    """Reads all CSV files in a specified folder and returns their contents as a dictionary.

    Args:
        folder_path (str): Path to the folder where the CSV files are stored.

    Returns:
        dict: A dictionary where keys are CSV file names and values are DataFrames containing
        the content of each CSV file.
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


def time_to_cross(dataframe, ids):
    """Calculates the time taken for each object with specified IDs to cross the road.

    Args:
        dataframe (DataFrame): The DataFrame containing object data.
        ids (list): A list of unique IDs of objects.

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
                        var[id] = count/30  # Assuming 30 frames per second
                        break

        else:
            for value in sorted_grp['X-center']:
                if value == x_max:
                    flag = 1
                if flag == 1:
                    count += 1
                    if value == x_min:
                        # Calculate time taken for crossing and store in dictionary
                        var[id] = count / 30  # Assuming 30 frames per second
                        break

    return var


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


def plot_scatter_diag(x, y, size, color, symbol,
                      city, plot_name, x_label, y_label,
                      legend_x=0.887, legend_y=0.986):
    """Plots a scatter plot with diagonal markers and annotations for city locations.

    Args:
        x (list): X-axis values.
        y (dict): Dictionary containing Y-axis values with city names as keys.
        size (list): Size of markers.
        color (list): Color of markers, representing continents.
        symbol (list): Symbol of markers, representing day/night.
        city (list): List of city names.
        plot_name (str): Name of the plot.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        legend_x (float, optional): X-coordinate for the legend. Defaults to 0.887.
        legend_y (float, optional): Y-coordinate for the legend. Defaults to 0.986.
    """
    # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green',
                        'Africa': 'red', 'North America': 'orange',
                        'South America': 'purple', 'Australia': 'brown'}

    # Create the scatter plot
    fig = px.scatter(x=x, y=list(y.values()), size=size, color=color,
                     symbol=symbol,  # Use conditions for symbols
                     labels={"color": "Continent"},  # Rename color legend
                     color_discrete_map=continent_colors)

    # Hide legend for all traces generated by Plotly Express
    for trace in fig.data:
        trace.showlegend = False

    # Adding labels and title
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

    # Add markers for continents
    for continent, color_ in continent_colors.items():
        if continent in color:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                     marker=dict(color=color_), name=continent))

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol,
                                             color='rgba(0,0,0,0)',
                                             line=dict(color='black', width=2)),
                                 name=description))

    # Adding annotations for locations
    annotations = []
    for i, key in enumerate(y.keys()):
        annotations.append(
            dict(x=x[i], y=list(y.values())[i], text=city[i], showarrow=False)
        )

    # Adjust annotation positions to avoid overlap
    adjusted_annotations = adjust_annotation_positions(annotations)
    fig.update_layout(annotations=adjusted_annotations)

    # Set template
    fig.update_layout(template=template)

    # Remove legend title
    fig.update_layout(legend_title_text='')

    # Update legend position
    fig.update_layout(
        legend=dict(x=legend_x, y=legend_y, traceorder="normal",))

    # Show the plot
    fig.show()

    # Save the plot
    save_plotly_figure(fig, plot_name)


def find_values(df, key):
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
        gdp = row["gdp_per_capita"]
        population = row["population"]
        population_country = row["population_country"]
        traffic_mortality = row["traffic_mortality"]
        continent = row["continent"]
        literacy_rate = row["literacy_rate"]
        avg_height = row["avg_height"]

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
                        return (video, s, end[counter], time_of_day_[counter], city, country, gdp, population,
                                population_country, traffic_mortality, continent, literacy_rate, avg_height)
                    counter += 1


def plot_cell_phone_vs_traffic_mortality(df_mapping, dfs):
    """Plots the relationship between cell phone usage and traffic mortality.

    Args:
        df_mapping (DataFrame): DataFrame containing mapping information.
        dfs (dict): Dictionary of DataFrames containing video data.
    """
    info = {}
    time_, traffic_mortality, continents,  = [], [], []
    gdp, conditions, city_ = [], [], []
    for key, value in dfs.items():
        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        # Count the number of mobile objects in the video
        mobile_ids = count_object(value, 67)

        # Calculate the duration of the video
        duration = end - start
        time_.append(duration)

        # Count the number of people in the video
        num_person = count_object(value, 0)

        # Extract the time of day
        condition = time_of_day

        # Calculate average cell phones detected per person
        if num_person == 0:
            continue
        avg_cell_phone = (((mobile_ids * 60) / time_[-1]) / num_person) * 1000

        # Update the information dictionary
        if f"{city}_{condition}" in info:
            previous_value = info[f"{city}_{condition}"]
            info[f"{city}_{condition}"] = (previous_value + avg_cell_phone) / 2
            # No need to add variables like traffic mortality, continents,
            # and so on again for the same city and condition
            continue
        else:
            info[f"{city}_{condition}"] = avg_cell_phone

        # Store additional information for plotting
        traffic_mortality.append(traffic_mortality_)
        continents.append(continent)
        gdp.append(gdp_)
        city_.append(city)
        conditions.append(condition)

    # Filter out values where info[key] == 0
    filtered_info = {k: v for k, v in info.items() if v != 0}
    filtered_traffic_mortality = [d for i, d in enumerate(traffic_mortality)
                                  if info[list(info.keys())[i]] != 0]
    filtered_continents = [c for i, c in enumerate(continents)
                           if info[list(info.keys())[i]] != 0]
    filtered_gdp = [c for i, c in enumerate(gdp)
                    if info[list(info.keys())[i]] != 0]
    filtered_conditions = [c for i, c in enumerate(conditions)
                           if info[list(info.keys())[i]] != 0]
    filtered_city = [c for i, c in enumerate(city_)
                     if info[list(info.keys())[i]] != 0]

    # Plot the scatter diagram
    plot_scatter_diag(x=filtered_traffic_mortality,
                      y=(filtered_info), size=filtered_gdp,
                      color=filtered_continents, symbol=filtered_conditions,
                      city=filtered_city,
                      plot_name="cell_phone_vs_traffic_mortality",
                      x_label="Traffic mortality rate per 100k person",
                      y_label="Number of Mobile detected in the video (normalised)",
                      legend_x=0, legend_y=0.986)


def plot_vehicle_vs_cross_time(df_mapping, dfs, data, motorcycle=1, car=1, bus=1, truck=1):
    """Plots the relationship between vehicle detection and crossing time.

    Args:
        df_mapping (DataFrame): DataFrame containing mapping information.
        dfs (dict): Dictionary of DataFrames containing video data.
        data (dict): Dictionary containing additional data.
        motorcycle (int, optional): Flag to include motorcycles. Default is 1.
        car (int, optional): Flag to include cars. Default is 1.
        bus (int, optional): Flag to include buses. Default is 1.
        truck (int, optional): Flag to include trucks. Default is 1.
    """

    info, time_dict = {}, {}
    time_avg, continents, gdp = [], [], []
    conditions, time_, city_ = [], [], []

    # Iterate through each video DataFrame
    for key, value in dfs.items():
        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        # Calculate the duration of the video
        duration = end - start
        time_.append(duration)

        time_cross = []
        dataframe = value

        # Extract the time of day
        condition = time_of_day

        # Filter vehicles based on flags
        if motorcycle == 1 & car == 1 & bus == 1 & truck == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 2) | (dataframe["YOLO_id"] == 3) |
                                    (dataframe["YOLO_id"] == 5) | (dataframe["YOLO_id"] == 7)]
            save_as = "all_vehicle_vs_cross_time"

        elif motorcycle == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 2)]
            save_as = "motorcycle_vs_cross_time"

        elif car == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 3)]
            save_as = "car_vs_cross_time"

        elif bus == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 5)]
            save_as = "bus_vs_cross_time"

        elif truck == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 7)]
            save_as = "truck_vs_cross_time"

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

        # Extract additional data for plotting
        time_dict = data[key]
        for key_, value in time_dict.items():
            time_cross.append(value)

        if not time_cross:
            info.popitem()
            time_.pop()
            continue

        time_avg.append(mean(time_cross))
        continents.append(continent)
        gdp.append(gdp_)
        city_.append(city)
        conditions.append(condition)

    # Plot the scatter diagram
    plot_scatter_diag(x=time_avg, y=info, size=gdp, color=continents,
                      symbol=conditions, city=city_, plot_name=save_as,
                      x_label="Average crossing time (in seconds)",
                      y_label="Number of vehicle detected (normalised)",
                      legend_x=0.887, legend_y=0.986)


def plot_time_to_start_crossing(dfs, person_id=0):
    """Plots the time taken by pedestrians to start crossing the road.

    Args:
        dfs (dict): Dictionary containing DataFrames of pedestrian data.
        person_id (int, optional): ID representing pedestrians. Default is 0.
    """
    time_dict, sd_dict, all_data = {}, {}, {}

    # Iterate through each video DataFrame
    for key, df in dfs.items():

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        data = {}
        crossed_ids = df[(df["YOLO_id"] == person_id)]

        # Makes group based on Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

        # Extract the time of day
        condition = time_of_day

        for unique_id, group_data in crossed_ids_grouped:
            x_values = group_data["X-center"].values
            initial_x = x_values[0]  # Initial x-value
            mean_height = group_data['Height'].mean()
            flag = 0
            margin = 0.1 * mean_height
            consecutive_frame = 0

            for i in range(0, len(x_values)-1):
                if initial_x < 0.5:
                    if (x_values[i] - margin <= x_values[i+1] <= x_values[i] + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            flag = 1
                    elif flag == 1:
                        data[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

                else:
                    if (x_values[i] - margin >= x_values[i+1] >= x_values[i] + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            flag = 1
                    elif flag == 1:
                        data[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

        if len(data) == 0:
            continue

        city_condition_key = f'{city}_{condition}'

        # Check if the city_condition combination already exists in all_data
        if city_condition_key in all_data:
            # Append the list of values from data to all_data
            all_data[city_condition_key].extend(data.values())
        else:
            # Add new entry to all_data
            all_data[city_condition_key] = list(data.values())

        # Calculate time_dict and sd_dict
        if all_data[city_condition_key]:
            time_dict[city_condition_key] = statistics.mean(all_data[city_condition_key]) / 30
            divided_data = [value / 30 for value in all_data[city_condition_key]]

            # Calculate the standard deviation of the divided data
            if len(divided_data) > 1:
                sd_dict[city_condition_key] = statistics.stdev(divided_data)
            else:
                sd_dict[city_condition_key] = 0  # Handle cases with less than 2 points
        else:
            # Handle the case when there are no values for the city_condition_key
            time_dict[city_condition_key] = 0
            sd_dict[city_condition_key] = 0

    day_values = {}
    night_values = {}
    for key, value in time_dict.items():
        city, condition = key.split('_')
        if city in day_values:
            if condition == '0':
                day_values[city] += value
            else:
                night_values[city] = value
        else:
            if condition == '0':
                day_values[city] = value
            else:
                night_values[city] = value

    # Fill missing values with 0
    for city in day_values.keys() | night_values.keys():
        day_values.setdefault(city, 0)
        night_values.setdefault(city, 0)

    # Sort data based on values for condition 0
    sorted_day_values = dict(sorted(
        day_values.items(), key=lambda item: item[1]))

    sorted_night_values = dict(sorted(
        night_values.items(), key=lambda item: item[1]))

    sorted_cities = list(sorted_day_values.keys())

    text_x = [0] * len(sorted_cities)

    # Create traces for condition 0
    trace_pos = go.Bar(
        x=list(sorted_day_values.values()),
        y=sorted_cities,
        orientation='h',
        name='Day',
        marker=dict(color='rgba(50, 171, 96, 0.6)')
    )

    # Create traces for condition 1
    trace_neg = go.Bar(
        x=[-night_values[city] for city in sorted_cities],
        y=sorted_cities,
        orientation='h',
        name='Night',
        marker=dict(color='rgba(219, 64, 82, 0.6)')
    )

    # Create figure
    fig = go.Figure(data=[trace_pos, trace_neg])

    # Update layout to include text labels
    max_value = int(max(max(day_values.values()), max(night_values.values())))
    fig.update_layout(
        # title='Double-Sided Bar Plot',
        barmode='relative',
        bargap=0.1,
        yaxis=dict(
            tickvals=[],
        ),
        xaxis=dict(
            title="Average time taken by the pedestrian to start crossing the road (in seconds)",
            tickvals=[-val for val in range(1, max_value + 1)] + [val for val in range(1, max_value + 1)],
            ticktext=[abs(val) for val in range(1, max_value + 1)] + [val for val in range(1, max_value + 1)]
        )
    )

    # Add text labels in the middle of each bar
    for i, city in enumerate(sorted_cities):
        fig.add_annotation(
            x=text_x[i],  # Set the x-coordinate of the text label
            y=city,  # Set the y-coordinate of the text label
            text=city,  # Set the text of the label to the city name
            font=dict(color='black'),  # Set text color to black
            showarrow=False,  # Do not show an arrow
            xanchor='center',  # Center the text horizontally
            yanchor='middle'  # Center the text vertically
        )

    for city in sorted_cities:
        if f"{city}_0" not in sd_dict:
            continue
        sd_value = "{:.3f}".format(sd_dict[f"{city}_0"])
        bar_value = sorted_day_values[city]
        x_coordinate = bar_value  # Set x-coordinate to the value of the bar
        mean_ = "{:.3f}".format(bar_value)
        fig.add_annotation(
            x=(x_coordinate/2) + 0.25,
            y=city,
            text=f"M={mean_}; SD={sd_value}",
            font=dict(color='black'),
            showarrow=False,
            xanchor='center',
            yanchor='middle'
        )

    for city in sorted_cities:
        if f"{city}_1" not in sd_dict:
            continue
        sd_value = "{:.3f}".format(sd_dict[f"{city}_1"])
        bar_value = sorted_night_values[city]
        x_coordinate = bar_value  # Set x-coordinate to the value of the bar
        mean_ = "{:.3f}".format(bar_value)
        fig.add_annotation(
            x=(-x_coordinate/2) - 0.25,
            y=city,
            text=f"M:{mean_}; SD:{sd_value}",
            font=dict(color='black'),
            showarrow=False,
            xanchor='center',
            yanchor='middle'
        )

    # Plot the figure
    fig.show()

    save_plotly_figure(fig, "time_to_start_cross")


def plot_no_of_pedestrian_stop(dfs, person_id=0):
    """Plots the number of pedestrian stops.

    Args:
        dfs (dict): Dictionary containing DataFrames of pedestrian data.
        person_id (int, optional): ID representing pedestrians. Default is 0.
    """
    count_dict = {}

    # Iterate through each video DataFrame
    for key, df in dfs.items():

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        data = {}
        count = 0

        # Extract the time of day
        condition = time_of_day

        crossed_ids = df[(df["YOLO_id"] == person_id)]

        # Makes group based on Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

        # Loop through each group to detect pedestrian stops
        for unique_id, group_data in crossed_ids_grouped:
            x_values = group_data["X-center"].values
            initial_x = x_values[0]  # Initial x-value
            mean_height = group_data['Height'].mean()
            flag = 0
            margin = 0.1 * mean_height
            consecutive_frame = 0

            # Check for pedestrian stops in each frame
            for i in range(0, len(x_values)-1):
                if initial_x < 0.5:
                    if (x_values[i] - margin <= x_values[i+1] <= x_values[i] + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            count += 1  # Increment count for each stop
                            flag = 1
                    elif flag == 1:
                        data[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

                else:
                    if (x_values[i] - margin >= x_values[i+1] >= x_values[i] + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            count += 1  # Increment count for each stop
                            flag = 1
                    elif flag == 1:
                        data[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

        if len(data) == 0:
            continue

        # Store the count in count_dict, normalizing by 1000
        if f'{city}_{condition}' in count_dict:
            old_count = count_dict[f'{city}_{condition}']
            new_count = old_count + (count/1000)
            count_dict[f'{city}_{condition}'] = new_count
            continue
        else:
            count_dict[f'{city}_{condition}'] = count/1000

    # Separate values into day and night categories
    day_values = {}
    night_values = {}
    for key, value in count_dict.items():
        city, condition = key.split('_')
        if city in day_values:
            if condition == '0':
                day_values[city] += value
            else:
                night_values[city] = value
        else:
            if condition == '0':
                day_values[city] = value
            else:
                night_values[city] = value

    # Fill missing values with 0
    for city in day_values.keys() | night_values.keys():
        day_values.setdefault(city, 0)
        night_values.setdefault(city, 0)

    # Sort data based on values for condition 0
    sorted_day_values = dict(sorted(
        day_values.items(), key=lambda item: item[1]))
    sorted_night_values = dict(sorted(
        night_values.items(), key=lambda item: item[1]))
    sorted_cities = list(sorted_day_values.keys())

    text_x = [0] * len(sorted_cities)

    # Create traces for condition 0
    trace_pos = go.Bar(
        x=list(sorted_day_values.values()),
        y=sorted_cities,
        orientation='h',
        name='Day',
        marker=dict(color='rgba(50, 171, 96, 0.6)')
    )

    # Create traces for condition 1
    trace_neg = go.Bar(
        x=[-night_values[city] for city in sorted_cities],
        y=sorted_cities,
        orientation='h',
        name='Night',
        marker=dict(color='rgba(219, 64, 82, 0.6)')
    )

    # Create figure
    fig = go.Figure(data=[trace_pos, trace_neg])

    # Update layout to include text labels
    max_value = int(max(max(day_values.values()), max(night_values.values())))
    fig.update_layout(
        # title='Double-Sided Bar Plot',
        barmode='relative',
        bargap=0.1,
        yaxis=dict(
            tickvals=[],
        ),
        xaxis=dict(
            title="No of pedestrian in the study (in thousands)",
            tickvals=[-val for val in range(1, max_value + 1)] + [val for val in range(1, max_value + 1)],
            ticktext=[abs(val) for val in range(1, max_value + 1)] + [val for val in range(1, max_value + 1)]
        )
    )

    # Add text labels in the middle of each bar
    for i, city in enumerate(sorted_cities):
        fig.add_annotation(
            x=text_x[i],  # Set the x-coordinate of the text label
            y=city,  # Set the y-coordinate of the text label
            text=city,  # Set the text of the label to the city name
            font=dict(color='black'),  # Set text color to black
            showarrow=False,  # Do not show an arrow
            xanchor='center',  # Center the text horizontally
            yanchor='middle'  # Center the text vertically
        )

    for city in sorted_cities:
        if f"{city}_0" not in count_dict:
            continue
        bar_value = sorted_day_values[city]
        x_coordinate = bar_value  # Set x-coordinate to the value of the bar
        bar_value_ = "{:.3f}".format(bar_value)
        bar_value_ = str(int(float(bar_value_) * 1000))
        fig.add_annotation(
            x=(x_coordinate/2) + 0.25,
            y=city,
            text=f"{bar_value_}",
            font=dict(color='black'),
            showarrow=False,
            xanchor='center',
            yanchor='middle'
        )

    for city in sorted_cities:
        if f"{city}_1" not in count_dict:
            continue
        bar_value = sorted_night_values[city]
        x_coordinate = bar_value  # Set x-coordinate to the value of the bar
        bar_value_ = "{:.3f}".format(bar_value)
        bar_value_ = str(int(float(bar_value_) * 1000))
        fig.add_annotation(
            x=(-x_coordinate/2) - 0.25,
            y=city,
            text=f"{bar_value_}",
            font=dict(color='black'),
            showarrow=False,
            xanchor='center',
            yanchor='middle'
        )

    # Plot the figure
    fig.show()

    save_plotly_figure(fig, "no_of_cases_for_cross")


def plot_hesitation_vs_traffic_mortality(df_mapping, dfs, person_id=0):
    """Plots the hesitation of pedestrians vs traffic mortality rate.

    Args:
        df_mapping (dict): Mapping of video keys to relevant information.
        dfs (dict): Dictionary of DataFrames containing pedestrian data.
        person_id (int, optional): ID representing pedestrians. Default is 0.
    """
    count_dict = {}
    time_, traffic_mortality, continents, gdp, conditions, city_ = [], [], [], [], [], []

    # Iterate through each video DataFrame
    for key, df in dfs.items():

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        count, pedestrian_count = 0, 0
        crossed_ids = df[(df["YOLO_id"] == person_id)]

        # Extract the time of day
        condition = time_of_day

        # Makes group based on Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

        for unique_id, group_data in crossed_ids_grouped:
            x_values = group_data["X-center"].values
            initial_x = x_values[0]  # Initial x-value
            consecutive_frames = 0

            # Check if initial x-value is less than 0.5
            if initial_x < 0.5:  # Check for crossing from left to right
                for i in range(0, len(x_values) - 10, 10):
                    # Check if x-value increases after every 10 frames
                    if (x_values[i + 10] > x_values[i]):
                        consecutive_frames += 1
                        if consecutive_frames >= 3:
                            pedestrian_count += 1
                            # Check if there's any instance where X-center [i + 1] <= X-center [i]
                            for j in range(i+1, len(x_values) - 10):
                                if x_values[j] <= x_values[j - 10]:
                                    count += 1
                                    break
                                break
            else:  # Check for crossing from right to left
                for i in range(0, len(x_values) - 10, 10):
                    # Check if x-value increases after every 10 frames
                    if (x_values[i + 10] < x_values[i]):
                        consecutive_frames += 1
                        if consecutive_frames >= 3:
                            pedestrian_count += 1
                            # Check if there's any instance where X-center [i + 1] >= X-center [i]
                            for j in range(i+1, len(x_values) - 10):
                                if x_values[j] >= x_values[j - 10]:
                                    count += 1
                                    break
                                break

        # Calculate the duration of the video
        duration = end - start
        time_.append(duration)

        # Calculate the percentage of people who hesitated while crossing
        if pedestrian_count == 0:
            continue
        count_ = ((((count * 60) * 100) / pedestrian_count) / duration)

        # Store the calculated value in count_dict
        if f'{city}_{condition}' in count_dict:
            old_count = count_dict[f'{city}_{condition}']
            new_count = old_count + count_
            count_dict[f'{city}_{condition}'] = new_count
            continue
        else:
            count_dict[f'{city}_{condition}'] = count_

        # Store additional data for plotting
        time_.append(duration)
        continents.append(continent)
        gdp.append(gdp_)
        city_.append(city)
        traffic_mortality.append(traffic_mortality_)
        conditions.append(condition)

    plot_scatter_diag(x=traffic_mortality, y=count_dict, size=gdp,
                      color=continents, symbol=conditions, city=city_,
                      plot_name="hesitation_vs_traffic_mortality",
                      x_label="Traffic mortality rate per 100k person",
                      y_label="Percentage of people who hesitated while crossing the road (normalised)",
                      legend_x=0.887, legend_y=0.986)


def plot_speed_of_crossing_vs_crossing_decision_time(df_mapping, dfs, data, person_id=0):
    """Plots speed of crossing vs crossing decision time.

    Args:
        df_mapping (dict): Mapping of video keys to relevant information.
        dfs (dict): Dictionary of DataFrames containing pedestrian data.
        data (dict): Dictionary containing crossing decision time data.
        person_id (int, optional): ID of the person to consider for crossing events. Defaults to 0.
    """
    avg_speed, time_dict, no_people = {}, {}, {}  # Dictionaries to store average speed, decision time, and counts
    continents, gdp, conditions, time_ = [], [], [], []  # Lists for continents, GDP, conditions, and time

    # Iterate over each video data
    for key, df in data.items():
        if df == {}:  # Skip if there is no data
            continue

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        value = dfs.get(key)

        # Extract the time of day
        condition = time_of_day
        length = avg_height

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

            ppm = mean_height / length
            distance = (max_x_center - min_x_center) / ppm

            speed_ = (distance / time) / 100
            if speed_ > 2.5:  # Exclude outlier speeds
                continue

            speed.append(speed_)

        no_people[f'{city}_{condition}'] = len(speed)

        if f'{city}_{condition}' in avg_speed:
            old_count = avg_speed[f'{city}_{condition}']
            new_count = old_count * no_people[f'{city}_{condition}'] + sum(speed)
            avg_speed[f'{city}_{condition}'] = new_count / (no_people[f'{city}_{condition}'] + len(speed))
        else:
            avg_speed[f'{city}_{condition}'] = sum(speed) / len(speed)

    for key, df in dfs.items():
        data, no_people = {}, {}
        crossed_ids = df[(df["YOLO_id"] == person_id)]

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        # Extract the time of day
        condition = time_of_day

        # Makes group based on Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

        for unique_id, group_data in crossed_ids_grouped:
            x_values = group_data["X-center"].values
            initial_x = x_values[0]  # Initial x-value
            mean_height = group_data['Height'].mean()
            flag = 0
            margin = 0.1 * mean_height  # Margin for considering crossing event
            consecutive_frame = 0

            for i in range(0, len(x_values)-1):
                if initial_x < 0.5:  # Check if crossing from left to right
                    if (x_values[i] - margin <= x_values[i+1] <= x_values[i] + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:  # Check for three consecutive frames
                            flag = 1
                    elif flag == 1:
                        data[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

                else:  # Check if crossing from right to left
                    if (x_values[i] - margin >= x_values[i+1] >= x_values[i] + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:  # Check for three consecutive frames
                            flag = 1
                    elif flag == 1:
                        data[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

        if len(data) == 0:  # Skip if no crossing events detected
            continue

        no_people[f'{city}_{condition}'] = len(data)

        if f'{city}_{condition}' in time_dict:
            old_count = time_dict[f'{city}_{condition}']
            new_count = old_count * no_people[f'{city}_{condition}'] + (sum(data.values()) / 30)
            time_dict[f'{city}_{condition}'] = new_count / (no_people[f'{city}_{condition}'] + len(data))  # noqa:E501
        else:
            time_dict[f'{city}_{condition}'] = ((sum(data.values()) / 30) / len(data))

    ordered_values = []

    for key in time_dict:
        city, condition = key.split('_')
        df_ = df_mapping[(df_mapping['city'] == city)]
        conditions.append(int(condition))
        continents.append(df_['continent'].values[0])
        gdp.append(df_['gdp_per_capita'].values[0])

        ordered_values.append((time_dict[key], avg_speed[key]))

    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange',
                        'South America': 'purple', 'Australia': 'brown'}

    fig = px.scatter(x=[value[0] for value in ordered_values],
                     y=[value[1] for value in ordered_values],
                     size=gdp,
                     color=continents,
                     symbol=conditions,  # Use conditions for symbols
                     labels={"color": "Continent"},  # Rename color legend
                     color_discrete_map=continent_colors)

    # Hide legend for all traces generated by Plotly Express
    for trace in fig.data:
        trace.showlegend = False

    # Adding labels and title
    fig.update_layout(
        xaxis_title="Average time pedestrian takes for crossing decision (in s)",
        yaxis_title="Average speed of pedestrian while crossing the road (in m/s)"
    )

    for continent, color in continent_colors.items():
        if continent in continents:
            fig.add_trace(go.Scatter(x=[None], y=[None],
                                     mode='markers', marker=dict(color=color),
                                     name=continent))

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)',
                                             line=dict(color='black', width=2)), name=description))

    # Adding annotations for locations
    annotations = []
    for i, key in enumerate(time_dict.keys()):
        location_name = key.split('_')[0]  # Extracting location name
        annotations.append(
            dict(
                x=[value[0] for value in ordered_values][i],
                y=[value[1] for value in ordered_values][i],
                text=location_name,  # Using location name instead of full key
                showarrow=False
            )
        )
    # Adjust annotation positions to avoid overlap
    adjusted_annotations = adjust_annotation_positions(annotations)
    fig.update_layout(annotations=adjusted_annotations)
    # set template
    fig.update_layout(template=template)

    # Remove legend title
    fig.update_layout(legend_title_text='')

    fig.update_layout(
            legend=dict(
                x=0.935,
                y=0.986,
                traceorder="normal",
            )
        )

    fig.show()
    save_plotly_figure(fig, "speed_of_crossing_vs_crossing_decision_time")


def plot_traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping, dfs, data):
    """Plots traffic mortality rate vs percentage of crossing events without traffic light.

    Args:
        df_mapping (dict): Mapping of video keys to relevant information.
        dfs (dict): Dictionary of DataFrames containing pedestrian data.
        data (dict): Dictionary containing pedestrian crossing data.
    """
    var_exist, var_nt_exist, ratio = {}, {}, {}
    continents, gdp, traffic_mortality = [], [], []
    conditions, time_, city_ = [], [], []

    # For a specific id of a person search for the first and last occurrence of that id and see if the traffic light
    # was present between it or not. Only getting those unique_id of the person who crosses the road.

    # Loop through each video data
    for key, df in data.items():
        counter_1, counter_2 = {}, {}
        counter_exists, counter_nt_exists = 0, 0

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

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
                                                (counter_1[f'{city}_{condition}'] + counter_2[f'{city}_{condition}']))
                continue
            # If already present, the array below will be filled multiple times
            else:
                ratio[f'{city}_{condition}'] = ((counter_2[f'{city}_{condition}'] * 100) /
                                                (counter_1[f'{city}_{condition}'] + counter_2[f'{city}_{condition}']))

        # Store additional data for plotting
        traffic_mortality.append(traffic_mortality_)
        continents.append(continent)
        gdp.append(gdp_)
        city_.append(city)
        conditions.append(condition)

    # Plot the scatter diagram
    plot_scatter_diag(x=traffic_mortality,
                      y=ratio, size=gdp,
                      color=continents, symbol=conditions,
                      city=city_,
                      plot_name="traffic_mortality_vs_crossing_event_wt_traffic_light",
                      x_label="Traffic mortality rate (per 100k)",
                      y_label="Percentage of Crossing Event without traffic light (normalised)",
                      legend_x=0, legend_y=0.986)


def plot_speed_of_crossing_vs_traffic_mortality(df_mapping, dfs, data):
    """Plots the average speed of crossing vs traffic mortality rate.

    Args:
        df_mapping (dict): Mapping of video keys to relevant information.
        dfs (dict): Dictionary of DataFrames containing pedestrian data.
        data (dict): Dictionary containing pedestrian crossing data.
    """
    avg_speed, no_people = {}, {}
    continents, gdp, traffic_mortality, = [], [], []
    conditions, time_, city_ = [], [], []

    # Loop through each video data
    for key, df in data.items():
        if df == {}:
            continue

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        value = dfs.get(key)

        length = avg_height

        # Calculate the duration of the video
        duration = end - start
        time_.append(duration)

        # Extract the time of day
        condition = time_of_day

        grouped = value.groupby('Unique Id')
        speed = []

        # Calculate speed for each pedestrian
        for id, time in df.items():
            grouped_with_id = grouped.get_group(id)
            mean_height = grouped_with_id['Height'].mean()
            min_x_center = grouped_with_id['X-center'].min()
            max_x_center = grouped_with_id['X-center'].max()

            ppm = mean_height / length
            distance = (max_x_center - min_x_center) / ppm

            # Calculate speed in meters per second (m/s)
            speed_ = (distance / time) / 100

            # Exclude unrealistic speeds (>2.5 m/s)
            if speed_ > 2.5:
                continue

            speed.append(speed_)

        # Store the number of pedestrians for each city and condition
        no_people[f'{city}_{condition}'] = len(speed)

        # Calculate the average speed for each city and condition
        if f'{city}_{condition}' in avg_speed:
            old_count = avg_speed[f'{city}_{condition}']
            new_count = old_count * no_people[f'{city}_{condition}'] + sum(speed)
            avg_speed[f'{city}_{condition}'] = new_count / (no_people[f'{city}_{condition}'] + len(speed))
            continue
        else:
            avg_speed[f'{city}_{condition}'] = sum(speed) / len(speed)

        # Store additional data for plotting
        time_.append(duration)
        traffic_mortality.append(traffic_mortality_)
        continents.append(continent)
        gdp.append(gdp_)
        city_.append(city)
        conditions.append(condition)

    # Plot the scatter diagram
    plot_scatter_diag(x=traffic_mortality,
                      y=avg_speed, size=gdp,
                      color=continents, symbol=conditions,
                      city=city_,
                      plot_name="speed_of_crossing_vs_traffic_mortality",
                      x_label="Traffic mortality rate (per 100k)",
                      y_label="Average speed of the pedestrian to cross the road (in m/s)",
                      legend_x=0, legend_y=0.986)


def plot_speed_of_crossing_vs_literacy(df_mapping, dfs, data):
    """Plots the average speed of crossing vs literacy rate.

    Args:
        df_mapping (dict): Mapping of video keys to relevant information.
        dfs (dict): Dictionary of DataFrames containing pedestrian data.
        data (dict): Dictionary containing pedestrian crossing data.
    """
    # Initialize dictionaries and lists to store data
    avg_speed, no_people = {}, {}  # to store average speed of crossing and number of pedestrians
    continents, gdp, literacy = [], [], []  # to store continent, GDP, and literacy rate
    conditions, time_, city_ = [], [], []  # to store conditions, time, and city

    # Loop through each video data
    for key, df in data.items():
        # Skip empty data
        if df == {}:
            continue

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        value = dfs.get(key)

        length = avg_height

        duration = end - start

        # Extract the time of day
        condition = time_of_day

        grouped = value.groupby('Unique Id')
        speed = []  # List to store pedestrian speeds

        for id, time in df.items():
            # Extract data for each pedestrian
            grouped_with_id = grouped.get_group(id)
            mean_height = grouped_with_id['Height'].mean()
            min_x_center = grouped_with_id['X-center'].min()
            max_x_center = grouped_with_id['X-center'].max()

            # Calculate pixels per meter (ppm) based on mean height
            ppm = mean_height / length
            # Calculate distance traversed by pedestrian
            distance = (max_x_center - min_x_center) / ppm
            # Calculate pedestrian speed
            speed_ = (distance / time) / 100
            # Skip pedestrians with unusually high speeds
            if speed_ > 2.5:
                continue

            speed.append(speed_)

        # Store the number of pedestrians for each condition
        no_people[f'{city}_{condition}'] = len(speed)

        # Calculate average speed for each condition
        if f'{city}_{condition}' in avg_speed:
            old_count = avg_speed[f'{city}_{condition}']
            new_count = old_count * no_people[f'{city}_{condition}'] + sum(speed)
            avg_speed[f'{city}_{condition}'] = new_count / (no_people[f'{city}_{condition}'] + len(speed))
            continue
        else:
            avg_speed[f'{city}_{condition}'] = sum(speed) / len(speed)

        # Store additional data for plotting
        time_.append(duration)
        conditions.append(condition)
        literacy.append(literacy_rate)
        continents.append(continent)
        gdp.append(gdp_)
        city_.append(city)

    # Plot the scatter diagram
    plot_scatter_diag(x=literacy,
                      y=avg_speed, size=gdp,
                      color=continents, symbol=conditions,
                      city=city_,
                      plot_name="speed_of_crossing_vs_literacy",
                      x_label="Literacy rate in the country (in percentage)",
                      y_label="Average speed of the pedestrian to cross the road (in m/s)",
                      legend_x=0, legend_y=1)


def plot_traffic_safety_vs_traffic_mortality(df_mapping, dfs):
    """Plots traffic safety vs traffic mortality.

    Args:
        df_mapping (dict): Mapping of video keys to relevant information.
        dfs (dict): Dictionary of DataFrames containing pedestrian data.
    """
    info, duration_ = {}, {}  # Dictionaries to store information and duration
    traffic_mortality, continents, gdp = [], [], []  # Lists for traffic mortality, continents, and GDP
    conditions, time_, city_ = [], [], []  # Lists for conditions, time, and city

    # Loop through each video data
    for key, value in dfs.items():

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

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

        # Store additional data for plotting
        traffic_mortality.append(traffic_mortality_)
        continents.append(continent)
        gdp.append(gdp_)
        city_.append(city)
        conditions.append(condition)
        time_.append(duration)

    # Plot the scatter diagram
    plot_scatter_diag(x=traffic_mortality,
                      y=info, size=gdp,
                      color=continents, symbol=conditions,
                      city=city_,
                      plot_name="traffic_safety_vs_traffic_mortality",
                      x_label="Traffic mortality rate per 100k person",
                      y_label="Number of traffic instruments detected (normalised)",
                      legend_x=0.887, legend_y=0.96)


def plot_traffic_safety_vs_literacy(df_mapping, dfs):
    """Plots traffic safety vs literacy.

    Args:
        df_mapping (dict): Mapping of video keys to relevant information.
        dfs (dict): Dictionary of DataFrames containing pedestrian data.
    """
    info, duration_ = {}, {}  # Dictionaries to store information and duration
    literacy, continents, gdp = [], [], []  # Lists for literacy, continents, and GDP
    conditions, time_, city_ = [], [], []  # Lists for conditions, time, and city

    # Loop through each video data
    for key, value in dfs.items():
        dataframe = value

        # Extract relevant information using the find_values function
        (_, start, end, time_of_day, city, country, gdp_, population,
         population_country, traffic_mortality_, continent, literacy_rate, avg_height) = find_values(df_mapping, key)

        duration = end - start
        condition = time_of_day

        # Filter dataframe for traffic instruments (YOLO_id 9 and 11)
        instrument = dataframe[(dataframe["YOLO_id"] == 9) | (dataframe["YOLO_id"] == 11)]

        # Skip if there are no instrument ids
        instrument_ids = instrument["Unique Id"].unique()

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

        # Store additional data for plotting
        continents.append(continent)
        gdp.append(gdp_)
        city_.append(city)
        conditions.append(condition)
        time_.append(duration)
        literacy.append(literacy_rate)

    # Plot the scatter diagram
    plot_scatter_diag(x=literacy, y=info, size=gdp, color=continents, symbol=conditions,
                      city=city_, plot_name="traffic_safety_vs_literacy",
                      x_label="Literacy rate in the country (in percentage)",
                      y_label="Number of traffic instruments detected (normalised)",
                      legend_x=0.07, legend_y=0.96)


# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")
    df_mapping = pd.read_csv("mapping.csv")
    dfs = read_csv_files(common.get_configs('data'))
    pedestrian_crossing_count, data = {}, {}

    # Loop over rows of data
    for key, value in dfs.items():
        logger.info("Analysing data from {}.", key)
        count, ids = pedestrian_crossing(dfs[key], 0.45, 0.55, 0)
        pedestrian_crossing_count[key] = {"count": count, "ids": ids}
        data[key] = time_to_cross(dfs[key], pedestrian_crossing_count[key]["ids"])

    # Data is dictionary in the form {Unique_id : Values}.
    # Values itself is another dictionary which is {Unique Id of person : Avg time to cross the road}
    # dfs is a dictionary in the form {Unique_id : CSV file}
    # df_mapping is the csv file

    plot_cell_phone_vs_traffic_mortality(df_mapping, dfs)
    plot_vehicle_vs_cross_time(df_mapping, dfs, data, motorcycle=1, car=1, bus=1, truck=1)
    plot_traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping, dfs, data)
    plot_hesitation_vs_traffic_mortality(df_mapping, dfs)

    plot_speed_of_crossing_vs_traffic_mortality(df_mapping, dfs, data)
    plot_speed_of_crossing_vs_literacy(df_mapping, dfs, data)

    plot_traffic_safety_vs_traffic_mortality(df_mapping, dfs)
    plot_traffic_safety_vs_literacy(df_mapping, dfs)

    plot_time_to_start_crossing(dfs)
    plot_no_of_pedestrian_stop(dfs)
    plot_speed_of_crossing_vs_crossing_decision_time(df_mapping, dfs, data)

    logger.info("Analysis completed.")
