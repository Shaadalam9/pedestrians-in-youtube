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


logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# set template for plotly output
template = common.get_configs('plotly_template')

# List of things that YOLO can detect:
# YOLO_id = {
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
#     9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
#     16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
#     24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
#     30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
#     36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
#     43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
#     51: 'carrot', 52: 'hot dog',53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',57: 'couch',
#     58: 'potted plant', 59: 'bed',60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
#     65: 'remote', 66: 'keyboard',67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
#     72: 'refrigerator', 73: 'book', 74: 'clock',75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
#     79: 'toothbrush'
# }


# Read the csv files and stores them as a dictionary in form {Unique_id : CSV}
def read_csv_files(folder_path):
    """Summary
    
    Args:
        folder_path (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    dfs = {}
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            filename = os.path.splitext(file)[0]
            dfs[filename] = df
    return dfs


def pedestrian_crossing(dataframe, min_x, max_x, person_id):
    """Summary
    
    Args:
        dataframe (TYPE): Description
        min_x (TYPE): Description
        max_x (TYPE): Description
        person_id (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    # sort only person in the dataframe
    crossed_ids = dataframe[(dataframe["YOLO_id"] == person_id)]
    # Makes group based on Unique ID
    crossed_ids_grouped = crossed_ids.groupby("Unique Id")
    filtered_crossed_ids = crossed_ids_grouped.filter(
        lambda x: (x["X-center"] <= min_x).any() and (x["X-center"] >= max_x).any())
    crossed_ids = filtered_crossed_ids["Unique Id"].unique()
    return len(crossed_ids), crossed_ids


def count_object(dataframe, id):
    """Summary
    
    Args:
        dataframe (TYPE): Description
        id (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    crossed_ids = dataframe[(dataframe["YOLO_id"] == id)]
    # Makes group based on Unique ID
    crossed_ids_grouped = crossed_ids.groupby("Unique Id")
    num_groups = crossed_ids_grouped.ngroups
    return num_groups


def time_to_cross(dataframe, ids):
    """Summary
    
    Args:
        dataframe (TYPE): Description
        ids (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    var = {}
    for id in ids:
        x_min = dataframe[dataframe["Unique Id"] == id]["X-center"].min()
        x_max = dataframe[dataframe["Unique Id"] == id]["X-center"].max()

        sorted_grp = dataframe[dataframe["Unique Id"] == id]

        x_min_index = sorted_grp[sorted_grp['X-center'] == x_min].index[0]
        x_max_index = sorted_grp[sorted_grp['X-center'] == x_max].index[0]

        count, flag = 0, 0
        if x_min_index < x_max_index:
            for value in sorted_grp['X-center']:
                if value == x_min:
                    flag = 1
                if flag == 1:
                    count += 1
                    if value == x_max:
                        var[id] = count/30
                        break

        else:
            for value in sorted_grp['X-center']:
                if value == x_max:
                    flag = 1
                if flag == 1:
                    count += 1
                    if value == x_min:
                        var[id] = count/30
                        break
    return var


def adjust_annotation_positions(annotations):
    """Summary
    
    Args:
        annotations (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    adjusted_annotations = []
    for i, ann in enumerate(annotations):
        adjusted_ann = ann.copy()
        # Adjust x and y coordinates to avoid overlap
        for other_ann in adjusted_annotations:
            if (abs(ann['x'] - other_ann['x']) < 0) and (abs(ann['y'] - other_ann['y']) < 0):
                adjusted_ann['y'] += 0
        adjusted_annotations.append(adjusted_ann)
    return adjusted_annotations


# makes scatter plots
def save_plotly_figure(fig, filename, width=1600, height=900, scale=3):
    """Summary
    
    Args:
        fig (TYPE): Description
        filename (TYPE): Description
        width (int, optional): Description
        height (int, optional): Description
        scale (int, optional): Description
    """
    # Create directory if it doesn't exist
    output_folder = "_output"
    os.makedirs(output_folder, exist_ok=True)

    # Save as HTML
    fig.write_html(os.path.join(output_folder, filename + ".html"))

    # Save as PNG
    fig.write_image(os.path.join(output_folder, filename + ".png"),
                    width=width,
                    height=height,
                    scale=scale)

    # Save as SVG
    fig.write_image(os.path.join(output_folder, filename + ".svg"),
                    format="svg")

    # Save as EPS
    fig.write_image(os.path.join(output_folder, filename + ".eps"),
                    width=width,
                    height=height)


def plot_scatter_diag(x, y, size, color, symbol,
                      city, plot_name, x_label, y_label,
                      legend_x=0.887, legend_y=0.986):
    """Summary
    
    Args:
        x (TYPE): Description
        y (TYPE): Description
        size (TYPE): Description
        color (TYPE): Description
        symbol (TYPE): Description
        city (TYPE): Description
        plot_name (TYPE): Description
        x_label (TYPE): Description
        y_label (TYPE): Description
        legend_x (float, optional): Description
        legend_y (float, optional): Description
    """
    # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green',
                        'Africa': 'red', 'North America': 'orange',
                        'South America': 'purple', 'Australia': 'brown'}

    fig = px.scatter(x=x,
                     y=list(y.values()),
                     size=size,
                     color=color,
                     symbol=symbol,  # Use conditions for symbols
                     labels={"color": "Continent"},  # Rename color legend
                     color_discrete_map=continent_colors)

    # Hide legend for all traces generated by Plotly Express
    for trace in fig.data:
        trace.showlegend = False

    # Adding labels and title
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    for continent, color_ in continent_colors.items():
        if continent in color:
            fig.add_trace(go.Scatter(x=[None], y=[None],
                                     mode='markers', marker=dict(color=color_),
                                     name=continent))

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                 mode='markers',
                                 marker=dict(symbol=symbol,
                                             color='rgba(0,0,0,0)',
                                 line=dict(color='black', width=2)),
                                 name=description))

    # Adding annotations for locations
    annotations = []
    for i, key in enumerate(y.keys()):
        annotations.append(
            dict(
                x=x[i],
                y=list(y.values())[i],
                text=city[i],
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
                x=legend_x,
                y=legend_y,
                traceorder="normal",
            )
        )

    fig.show()
    save_plotly_figure(fig, plot_name)


def get_duration_for_key(video_column, duration_column, key):
    """Summary
    
    Args:
        video_column (TYPE): Description
        duration_column (TYPE): Description
        key (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    for video, duration in zip(video_column, duration_column):
        if key in video:
            video_list = video.strip('[]').split(',')
            key_index = video_list.index(key)
            duration_list = duration.strip('[]').split(',')
            return int(duration_list[key_index])


def get_single_value_for_key(video_column, value_column, key):
    """Summary
    
    Args:
        video_column (TYPE): Description
        value_column (TYPE): Description
        key (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    for video, duration in zip(video_column, value_column):
        if key in video:
            value = duration  # Convert duration to integer
            return value


# Plotting Functions
def plot_cell_phone_vs_traffic_mortality(df_mapping, dfs):
    """Summary
    
    Args:
        df_mapping (TYPE): Description
        dfs (TYPE): Description
    """
    info = {}
    time_, traffic_mortality, continents,  = [], [], []
    gdp, conditions, city_ = [], [], []
    for key, value in dfs.items():

        mobile_ids = count_object(value, 67)

        duration = get_duration_for_key(df_mapping['videos'],
                                        df_mapping['duration'], key)
        time_.append(duration)

        num_person = count_object(value, 0)

        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], key)

        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], key))

        # Saving the number of unique mobile discovered in the video
        if num_person == 0:
            continue
        avg_cell_phone = (((mobile_ids * 60) / time_[-1]) / num_person) * 1000

        if f"{city}_{condition}" in info:
            previous_value = info[f"{city}_{condition}"]
            info[f"{city}_{condition}"] = (previous_value + avg_cell_phone) / 2
            # No need to add variables like traffic mortality, continents and son on again for same city and condition
            continue  

        else:
            info[f"{city}_{condition}"] = avg_cell_phone

        traffic_mortality.append(float(get_single_value_for_key(
            df_mapping['videos'], df_mapping['traffic_mortality'], key)))

        continents.append(get_single_value_for_key(
            df_mapping['videos'], df_mapping['continent'], key))

        gdp.append(int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['gdp_per_capita'], key)))

        city_.append(get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], key))
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

    plot_scatter_diag(x=filtered_traffic_mortality,
                      y=(filtered_info), size=filtered_gdp,
                      color=filtered_continents, symbol=filtered_conditions,
                      city=filtered_city,
                      plot_name="cell_phone_vs_traffic_mortality",
                      x_label="Traffic mortality rate per 100k person",
                      y_label="Number of Mobile detected in the video (normalised)",
                      legend_x=0, legend_y=0.986)


# TODO: check if there is a csv with avg vehicle ownership/usage on the city/country level 
def plot_vehicle_vs_cross_time(df_mapping, dfs, data, motorcycle=0, car=0, bus=0, truck=0):
    """Summary
    
    Args:
        df_mapping (TYPE): Description
        dfs (TYPE): Description
        data (TYPE): Description
        motorcycle (int, optional): Description
        car (int, optional): Description
        bus (int, optional): Description
        truck (int, optional): Description
    """
    info, time_dict = {}, {}
    time_avg, continents, gdp = [], [], []
    conditions, time_, city_ = [], [], []

    for key, value in dfs.items():
        duration = get_duration_for_key(df_mapping['videos'],
                                        df_mapping['duration'],
                                        key)
        time_.append(duration)
        time_cross = []
        dataframe = value

        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], key)

        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], key))

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
            print("No plot generated")

        vehicle_ids = vehicle_ids["Unique Id"].unique()

        if vehicle_ids is None:
            continue

        # contains {location : no of vehicle detected}
        new_value = ((len(vehicle_ids)/time_[-1]) * 60)
        # The detection is normalised, so both the values can be added directly
        if f"{city}_{condition}" in info:
            previous_value = info[f"{city}_{condition}"]
            info[f"{city}_{condition}"] = (previous_value + new_value) / 2
            continue

        else:
            info[f"{city}_{condition}"] = new_value

        time_dict = data[key]
        for key_, value in time_dict.items():
            time_cross.append(value)

        if not time_cross:
            info.popitem()
            time_.pop()
            continue

        time_avg.append(mean(time_cross))

        continents.append(get_single_value_for_key(
            df_mapping['videos'], df_mapping['continent'], key))

        gdp.append(int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['gdp_per_capita'], key)))

        city_.append(city)
        conditions.append(condition)

    plot_scatter_diag(x=time_avg, y=info, size=gdp, color=continents,
                      symbol=conditions, city=city_, plot_name=save_as,
                      x_label="Average crossing time (in seconds)",
                      y_label="Number of vehicle detected (normalised)",
                      legend_x=0.887, legend_y=0.986)


# TODO : Add the flag
# TODO : Check the calculation of mean and SD
# On an average how many times a person who is crossing a road will hesitate to do it. 
def plot_time_to_start_crossing(dfs, person_id=0):
    """Summary
    
    Args:
        dfs (TYPE): Description
        person_id (int, optional): Description
    """
    time_dict, sd_dict = {}, {}
    for location, df in dfs.items():
        data = {}
        crossed_ids = df[(df["YOLO_id"] == person_id)]

        # Makes group based on Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")
        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], location))
        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], location)

        # Initialize dictionaries to track sum and sum of squares for each city_condition combination 
        sum_values = {}
        sum_squares = {}

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

        values = [value / 30 for value in data.values()]
        sd = statistics.stdev(values)

        city_condition_key = f'{city}_{condition}'

        # Update sum and sum of squares for the city_condition combination
        sum_values[city_condition_key] = sum(data.values())
        sum_squares[city_condition_key] = sum(value**2 for value in data.values()) 

        # Check if the city_condition combination already exists in time_dict
        if city_condition_key in time_dict:
            # Adjust the mean and standard deviation
            old_mean = time_dict[city_condition_key]
            old_sd = sd_dict[city_condition_key]
            n_old = len(data) - 1  # Number of previous data points
            n_new = 1  # Number of new data points
            new_sum = sum_values[city_condition_key]
            new_sum_squares = sum_squares[city_condition_key]

            new_mean = (old_mean * n_old + new_sum) / (n_old + n_new)
            new_sd = ((old_sd ** 2 * n_old + new_sum_squares) / (n_old + n_new) - new_mean ** 2) ** 0.5 

            time_dict[city_condition_key] = new_mean
            sd_dict[city_condition_key] = new_sd
            continue
        else:
            # Add new entry to time_dict and sd_dict
            time_dict[city_condition_key] = sum(data.values()) / len(data) / 30
            sd_dict[city_condition_key] = sd

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


# TODO : Find the difference between a person as a pedestrian and as motorcyclist
def plot_no_of_pedestrian_stop(dfs, person_id=0):
    """Summary
    
    Args:
        dfs (TYPE): Description
        person_id (int, optional): Description
    """
    count_dict = {}
    for location, df in dfs.items():
        data = {}
        count = 0

        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], location))

        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], location)

        crossed_ids = df[(df["YOLO_id"] == person_id)]

        # Makes group based on Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

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
                            count += 1
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
                            count += 1
                            flag = 1
                    elif flag == 1:
                        data[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

        if len(data) == 0:
            continue

        if f'{city}_{condition}' in count_dict:
            old_count = count_dict[f'{city}_{condition}']
            new_count = old_count + (count/1000)
            count_dict[f'{city}_{condition}'] = new_count
            continue
        else:
            count_dict[f'{city}_{condition}'] = count/1000

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
        bar_value_ = str(float(bar_value_) * 1000)
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
        bar_value_ = str(float(bar_value_) * 1000)
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
    """Summary
    
    Args:
        df_mapping (TYPE): Description
        dfs (TYPE): Description
        person_id (int, optional): Description
    """
    count_dict = {}
    time_, traffic_mortality, continents, gdp, conditions, city_ = [], [], [], [], [], [] 
    for key, df in dfs.items():
        count, pedestrian_count = 0, 0
        crossed_ids = df[(df["YOLO_id"] == person_id)]
        city = get_single_value_for_key(df_mapping['videos'], df_mapping['city'], key) 
        condition = int(get_single_value_for_key(df_mapping['videos'], df_mapping['time_of_day'], key)) 

        # Makes group based on Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

        for unique_id, group_data in crossed_ids_grouped:
            x_values = group_data["X-center"].values
            initial_x = x_values[0]  # Initial x-value
            consecutive_frames = 0

            # Check if initial x-value is less than 0.5
            # Check for crossing from left to right
            if initial_x < 0.5:
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
            else:
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

        duration = get_duration_for_key(df_mapping['videos'], df_mapping['duration'], key)

        # num_person = count_object(df, 0)
        if pedestrian_count == 0:
            continue

        count_ = ((((count * 60) * 100) / pedestrian_count) / duration)

        if f'{city}_{condition}' in count_dict:
            old_count = count_dict[f'{city}_{condition}']
            new_count = old_count + count_
            count_dict[f'{city}_{condition}'] = new_count
            continue
        else:
            count_dict[f'{city}_{condition}'] = count_

        time_.append(duration)
        continents.append(get_single_value_for_key(df_mapping['videos'], df_mapping['continent'], key))
        gdp.append(int(get_single_value_for_key(df_mapping['videos'], df_mapping['gdp_per_capita'], key)))
        traffic_mortality.append(get_single_value_for_key(df_mapping['videos'], df_mapping['traffic_mortality'], key))
        city_.append(city)
        conditions.append(condition)

    plot_scatter_diag(x=traffic_mortality, y=count_dict, size=gdp,
                      color=continents, symbol=conditions, city=city_,
                      plot_name="hesitation_vs_traffic_mortality",
                      x_label="Traffic mortality rate per 100k person",
                      y_label="Percentage of people who hesitated while crossing the road (normalised)",
                      legend_x=0.887, legend_y=0.986)


def plot_speed_of_crossing_vs_crossing_decision_time(df_mapping, dfs, data, person_id=0):
    """Summary
    
    Args:
        df_mapping (TYPE): Description
        dfs (TYPE): Description
        data (TYPE): Description
        person_id (int, optional): Description
    """
    avg_speed, time_dict, no_people = {}, {}, {}
    continents, gdp, conditions, time_ = [], [], [], []

    for key, df in data.items():
        if df == {}:
            continue
        value = dfs.get(key)

        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], key)

        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], key))

        length = get_single_value_for_key(
            df_mapping['videos'], df_mapping['avg_height'], key)

        duration = get_duration_for_key(
            df_mapping['videos'], df_mapping['duration'], key)

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
            if speed_ > 2.5:
                continue

            speed.append(speed_)

        no_people[f'{city}_{condition}'] = len(speed)

        if f'{city}_{condition}' in avg_speed:
            old_count = avg_speed[f'{city}_{condition}']
            new_count = old_count * no_people[f'{city}_{condition}'] + sum(speed)
            avg_speed[f'{city}_{condition}'] = new_count / (no_people[f'{city}_{condition}'] + len(speed))
        else:
            avg_speed[f'{city}_{condition}'] = sum(speed) / len(speed)

    for location, df in dfs.items():
        data, no_people = {}, {}
        crossed_ids = df[(df["YOLO_id"] == person_id)]
        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], location)
        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], location))

        # Makes group based on Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")

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

        no_people[f'{city}_{condition}'] = len(data)

        if f'{city}_{condition}' in time_dict:
            old_count = time_dict[f'{city}_{condition}']
            new_count = old_count * no_people[f'{city}_{condition}'] + (sum(data.values()) / 30)
            time_dict[f'{city}_{condition}'] = new_count / (no_people[f'{city}_{condition}'] + len(data))
        else:
            time_dict[f'{city}_{condition}'] = ((sum(data.values()) / 30) / len(data))

    ordered_values = []
    for key in time_dict:
        city, condition = key.split('_')
        df_ = df_mapping[(df_mapping['city'] == city) & (df_mapping['time_of_day'] == int(condition))]
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
        fig.add_trace(go.Scatter(x=[None],
                                 y=[None], mode='markers',
                                 marker=dict(
                                     symbol=symbol,
                                     color='rgba(0,0,0,0)',
                                     line=dict(
                                         color='black',
                                         width=2)), name=description))

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
    """Summary
    
    Args:
        df_mapping (TYPE): Description
        dfs (TYPE): Description
        data (TYPE): Description
    """
    var_exist, var_nt_exist, ratio = {}, {}, {}
    continents, gdp, traffic_mortality = [], [], []
    conditions, time_, city_ = [], [], []

    # For a specific id of a person search for the first and last occurrence of that id and see if the traffic light
    # was present between it or not. Only getting those unique_id of the person who crosses the road.

    for key, df in data.items():
        counter_1, counter_2 = {}, {}
        counter_exists, counter_nt_exists = 0, 0

        time_.append(get_duration_for_key(
            df_mapping['videos'], df_mapping['duration'], key))

        value = dfs.get(key)
        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], key)

        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], key))

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

        traffic_mortality.append(float(
            get_single_value_for_key(df_mapping['videos'],
                                     df_mapping['traffic_mortality'], key)))

        continents.append(
            get_single_value_for_key(df_mapping['videos'],
                                     df_mapping['continent'], key))

        gdp.append(int(
            get_single_value_for_key(df_mapping['videos'],
                                     df_mapping['gdp_per_capita'], key)))

        city_.append(city)
        conditions.append(condition)

    plot_scatter_diag(x=traffic_mortality,
                      y=ratio, size=gdp,
                      color=continents, symbol=conditions,
                      city=city_,
                      plot_name="traffic_mortality_vs_crossing_event_wt_traffic_light",
                      x_label="Traffic mortality rate (per 100k)",
                      y_label="Percentage of Crossing Event without traffic light (normalised)",
                      legend_x=0, legend_y=0.986)


def plot_speed_of_crossing_vs_traffic_mortality(df_mapping, dfs, data):
    """Summary
    
    Args:
        df_mapping (TYPE): Description
        dfs (TYPE): Description
        data (TYPE): Description
    """
    avg_speed, no_people = {}, {}
    continents, gdp, traffic_mortality, = [], [], []
    conditions, time_, city_ = [], [], [] 
    for key, df in data.items():
        if df == {}:
            continue
        value = dfs.get(key)

        length = get_single_value_for_key(
            df_mapping['videos'], df_mapping['avg_height'], key)

        duration = int(get_duration_for_key(
            df_mapping['videos'], df_mapping['duration'], key))

        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], key))

        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], key)

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
            if speed_ > 2.5:
                continue

            speed.append(speed_)

        no_people[f'{city}_{condition}'] = len(speed)

        if f'{city}_{condition}' in avg_speed:
            old_count = avg_speed[f'{city}_{condition}']
            new_count = old_count * no_people[f'{city}_{condition}'] + sum(speed)
            avg_speed[f'{city}_{condition}'] = new_count / (no_people[f'{city}_{condition}'] + len(speed))
            continue
        else:
            avg_speed[f'{city}_{condition}'] = sum(speed) / len(speed)

        time_.append(duration)
        conditions.append(condition)

        traffic_mortality.append(float(get_single_value_for_key(
            df_mapping['videos'], df_mapping['traffic_mortality'], key)))

        continents.append(get_single_value_for_key(
            df_mapping['videos'], df_mapping['continent'], key))

        gdp.append(int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['gdp_per_capita'], key)))

        city_.append(city)

    plot_scatter_diag(x=traffic_mortality,
                      y=avg_speed, size=gdp,
                      color=continents, symbol=conditions,
                      city=city_,
                      plot_name="speed_of_crossing_vs_traffic_mortality",
                      x_label="Traffic mortality rate (per 100k)",
                      y_label="Average speed of the pedestrian to cross the road (in m/s)",
                      legend_x=0, legend_y=0.986)


def plot_speed_of_crossing_vs_literacy(df_mapping, dfs, data):
    """Summary
    
    Args:
        df_mapping (TYPE): Description
        dfs (TYPE): Description
        data (TYPE): Description
    """
    avg_speed, no_people = {}, {}
    continents, gdp, literacy = [], [], []
    conditions, time_, city_ = [], [], []
    for key, df in data.items():
        if df == {}:
            continue
        value = dfs.get(key)

        length = get_single_value_for_key(
            df_mapping['videos'], df_mapping['avg_height'], key)

        duration = int(get_duration_for_key(
            df_mapping['videos'], df_mapping['duration'], key))

        condition = int(get_single_value_for_key
                        (df_mapping['videos'], df_mapping['time_of_day'], key))

        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], key)

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
            if speed_ > 2.5:
                continue

            speed.append(speed_)

        no_people[f'{city}_{condition}'] = len(speed)

        if f'{city}_{condition}' in avg_speed:
            old_count = avg_speed[f'{city}_{condition}']
            new_count = old_count * no_people[f'{city}_{condition}'] + sum(speed)
            avg_speed[f'{city}_{condition}'] = new_count / (no_people[f'{city}_{condition}'] + len(speed))
            continue
        else:
            avg_speed[f'{city}_{condition}'] = sum(speed) / len(speed)

        time_.append(duration)
        conditions.append(condition)

        literacy.append(get_single_value_for_key(
            df_mapping['videos'], df_mapping['literacy_rate'], key))

        continents.append(get_single_value_for_key(
            df_mapping['videos'], df_mapping['continent'], key))

        gdp.append(int(get_single_value_for_key
                       (df_mapping['videos'], df_mapping['gdp_per_capita'],
                        key)))

        city_.append(city)

    plot_scatter_diag(x=literacy,
                      y=avg_speed, size=gdp,
                      color=continents, symbol=conditions,
                      city=city_,
                      plot_name="speed_of_crossing_vs_literacy",
                      x_label="Literacy rate in the country (in percentage)",
                      y_label="Average speed of the pedestrian to cross the road (in m/s)",
                      legend_x=0, legend_y=1)


def plot_traffic_safety_vs_traffic_mortality(df_mapping, dfs):
    """Summary
    
    Args:
        df_mapping (TYPE): Description
        dfs (TYPE): Description
    """
    info, duration_ = {}, {}
    traffic_mortality, continents, gdp = [], [], []
    conditions, time_, city_ = [], [], []

    for key, value in dfs.items():
        dataframe = value

        duration = int(get_duration_for_key(
            df_mapping['videos'], df_mapping['duration'], key))

        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], key)

        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], key))

        instrument = dataframe[(dataframe["YOLO_id"] == 9) | (dataframe["YOLO_id"] == 11)]

        instrument_ids = instrument["Unique Id"].unique()

        if instrument_ids is None:
            continue

        count_ = ((len(instrument_ids)/duration) * 60)

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

        traffic_mortality.append(float(get_single_value_for_key(
            df_mapping['videos'], df_mapping['traffic_mortality'], key)))

        continents.append(get_single_value_for_key(
            df_mapping['videos'], df_mapping['continent'], key))

        gdp.append(int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['gdp_per_capita'], key)))

        city_.append(city)
        conditions.append(condition)
        time_.append(duration)

    plot_scatter_diag(x=traffic_mortality,
                      y=info, size=gdp,
                      color=continents, symbol=conditions,
                      city=city_,
                      plot_name="traffic_safety_vs_traffic_mortality",
                      x_label="Traffic mortality rate per 100k person",
                      y_label="Number of traffic instruments detected (normalised)",
                      legend_x=0.887, legend_y=0.96)


def plot_traffic_safety_vs_literacy(df_mapping, dfs):
    """Summary
    
    Args:
        df_mapping (TYPE): Description
        dfs (TYPE): Description
    """
    info, duration_ = {}, {}
    literacy, continents, gdp = [], [], []
    conditions, time_, city_ = [], [], []
    for key, value in dfs.items():
        dataframe = value

        duration = int(get_duration_for_key(
            df_mapping['videos'], df_mapping['duration'], key))

        city = get_single_value_for_key(
            df_mapping['videos'], df_mapping['city'], key)

        condition = int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['time_of_day'], key))

        instrument = dataframe[(dataframe["YOLO_id"] == 9) | (dataframe["YOLO_id"] == 11)]

        instrument_ids = instrument["Unique Id"].unique()

        if instrument_ids is None:
            continue

        count_ = ((len(instrument_ids)/duration) * 60)

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

        continents.append(get_single_value_for_key(
            df_mapping['videos'], df_mapping['continent'], key))

        gdp.append(int(get_single_value_for_key(
            df_mapping['videos'], df_mapping['gdp_per_capita'], key)))

        city_.append(city)
        conditions.append(condition)
        time_.append(duration)
        literacy.append(get_single_value_for_key(
            df_mapping['videos'], df_mapping['literacy_rate'], key))

    plot_scatter_diag(x=literacy,
                      y=info, size=gdp,
                      color=continents, symbol=conditions,
                      city=city_,
                      plot_name="traffic_safety_vs_literacy",
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

    # Data is dictionary in the form {Unique_id : Values}.Values itself
    # is another dictionary which is {Unique Id of person : Avg time to cross
    # the road}. dfs is a dictionary in the form {Unique_id : CSV file}
    # df_mapping is the csv file
    plot_cell_phone_vs_traffic_mortality(df_mapping, dfs)
    plot_vehicle_vs_cross_time(df_mapping,
                               dfs,
                               data,
                               motorcycle=1,
                               car=1,
                               bus=1,
                               truck=1)
    plot_traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping,
                                                              dfs,
                                                              data)
    plot_hesitation_vs_traffic_mortality(df_mapping, dfs)

    plot_speed_of_crossing_vs_traffic_mortality(df_mapping, dfs, data)
    plot_speed_of_crossing_vs_literacy(df_mapping, dfs, data)

    plot_traffic_safety_vs_traffic_mortality(df_mapping, dfs)
    plot_traffic_safety_vs_literacy(df_mapping, dfs)

    plot_time_to_start_crossing(dfs)
    plot_no_of_pedestrian_stop(dfs)
    plot_speed_of_crossing_vs_crossing_decision_time(df_mapping, dfs, data)

    logger.info("Analysis completed.")
