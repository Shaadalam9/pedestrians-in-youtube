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
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',  # noqa: E501
#     9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',  # noqa: E501
#     16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',  # noqa: E501
#     25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',  # noqa: E501
#     33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',  # noqa: E501
#     40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',  # noqa: E501
#     49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',57: 'couch',  # noqa: E501
#     58: 'potted plant', 59: 'bed',60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',  # noqa: E501
#     66: 'keyboard',67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',  # noqa: E501
#     74: 'clock',75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'  # noqa: E501
# }


def read_csv_files(folder_path):
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
    # sort only person in the dataframe
    crossed_ids = dataframe[(dataframe["YOLO_id"] == person_id)]
    # Makes group based on Unique ID
    crossed_ids_grouped = crossed_ids.groupby("Unique Id")
    filtered_crossed_ids = crossed_ids_grouped.filter(
        lambda x: (x["X-center"] <= min_x).any() and (x["X-center"] >= max_x).any())  # noqa: E501
    crossed_ids = filtered_crossed_ids["Unique Id"].unique()
    return len(crossed_ids), crossed_ids


def count_object(dataframe, id):
    crossed_ids = dataframe[(dataframe["YOLO_id"] == id)]
    # Makes group based on Unique ID
    crossed_ids_grouped = crossed_ids.groupby("Unique Id")
    num_groups = crossed_ids_grouped.ngroups
    return num_groups


def time_to_cross(dataframe, ids):
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
    adjusted_annotations = []
    for i, ann in enumerate(annotations):
        adjusted_ann = ann.copy()
        # Adjust x and y coordinates to avoid overlap
        for other_ann in adjusted_annotations:
            if (abs(ann['x'] - other_ann['x']) < 0) and (abs(ann['y'] - other_ann['y']) < 0):  # noqa: E501
                adjusted_ann['y'] += 0
        adjusted_annotations.append(adjusted_ann)
    return adjusted_annotations


def save_plotly_figure(fig, filename_html, filename_png, filename_svg, width=1600, height=900, scale=3):  # noqa: E501
    # Create directory if it doesn't exist
    output_folder = "_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Save as HTML
    fig.write_html(os.path.join(output_folder, filename_html))

    # Save as PNG
    fig.write_image(os.path.join(output_folder, filename_png),
                    width=width, height=height, scale=scale)

    # Save as SVG
    fig.write_image(os.path.join(output_folder, filename_svg),
                    format="svg")


def plot_scatter_diag(x, y, size, color, symbol, city, plot_name, x_label, y_label, legend_x=0.887, legend_y=0.986):  # noqa: E501
    # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

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
        xaxis_title=x_label,  # noqa: E501
        yaxis_title=y_label
    )

    for continent, color_ in continent_colors.items():
        if continent in color:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color_), name=continent))  # noqa: E501

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)', line=dict(color='black', width=2)), name=description))  # noqa: E501

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
    save_plotly_figure(fig, f"{plot_name}.html", f"{plot_name}.png", f"{plot_name}.svg")  # noqa: E501


def get_duration_for_key(video_column, duration_column, key):
    for video, duration in zip(video_column, duration_column):
        if key in video:
            video_list = video.strip('[]').split(',')
            key_index = video_list.index(key)
            duration_list = duration.strip('[]').split(',')
            return int(duration_list[key_index])


def get_single_value_for_key(video_column, value_column, key):
    for video, duration in zip(video_column, value_column):
        if key in video:
            value = duration  # Convert duration to integer
            return value


# Plotting Functions
def plot_cell_phone_vs_traffic_mortality(df_mapping, dfs):
    info = {}
    time_, traffic_mortality, continents, gdp, conditions, city_ = [], [], [], [], [], []  # noqa: E501
    for key, value in dfs.items():

        mobile_ids = count_object(value, 67)

        duration = get_duration_for_key(df_mapping['videos'], df_mapping['duration'], key)  # noqa: E501
        time_.append(duration)

        num_person = count_object(value, 0)

        city = get_single_value_for_key(df_mapping['videos'], df_mapping['city'], key)  # noqa: E501
        condition = int(get_single_value_for_key(df_mapping['videos'], df_mapping['time_of_day'], key))  # noqa: E501

        # Saving the number of unique mobile discovered in the video
        avg_cell_phone = (((mobile_ids * 60) / time_[-1]) / num_person)

        if f"{city}_{condition}" in info:
            previous_value = info[key]
            info[key] = (previous_value + avg_cell_phone) / 2

        else:
            info[key] = avg_cell_phone * 1000

        traffic_mortality.append(float(get_single_value_for_key(df_mapping['videos'], df_mapping['traffic_mortality'], key)))  # noqa: E501
        continents.append(get_single_value_for_key(df_mapping['videos'], df_mapping['continent'], key))  # noqa: E501
        gdp.append(int(get_single_value_for_key(df_mapping['videos'], df_mapping['gdp_per_capita'], key)))  # noqa: E501
        city_.append(get_single_value_for_key(df_mapping['videos'], df_mapping['city'], key))  # noqa: E501
        conditions.append(condition)

    # Filter out values where info[key] == 0
    filtered_info = {k: v for k, v in info.items() if v != 0}
    filtered_traffic_mortality = [d for i, d in enumerate(traffic_mortality) if info[list(info.keys())[i]] != 0]  # noqa: E501
    filtered_continents = [c for i, c in enumerate(continents) if info[list(info.keys())[i]] != 0]  # noqa: E501
    filtered_gdp = [c for i, c in enumerate(gdp) if info[list(info.keys())[i]] != 0]   # noqa: E501
    filtered_conditions = [c for i, c in enumerate(conditions) if info[list(info.keys())[i]] != 0]   # noqa: E501
    filtered_city = [c for i, c in enumerate(city_) if info[list(info.keys())[i]] != 0]   # noqa: E501

    plot_scatter_diag(x=filtered_traffic_mortality,
                      y=(filtered_info), size=filtered_gdp,
                      color=filtered_continents, symbol=filtered_conditions,
                      city=filtered_city,
                      plot_name="cell_phone_vs_traffic_mortality",
                      x_label="Traffic mortality rate per 100k person",
                      y_label="Number of Mobile detected in the video (normalised)",  # noqa: E501
                      legend_x=0, legend_y=0.986)


# TODO: check if there is a csv with avg vehicle ownership/usage on the city/country level   # noqa: E501
def plot_vehicle_vs_cross_time(df_mapping, dfs, data, motorcycle=0, car=0, bus=0, truck=0):    # noqa: E501
    info, time_dict = {}, {}
    time_avg, continents, gdp, conditions, time_, city_ = [], [], [], [], [], []  # noqa: E501

    for key, value in dfs.items():
        duration = get_duration_for_key(df_mapping['videos'], df_mapping['duration'], key)  # noqa: E501
        time_.append(duration)
        time_cross = []
        dataframe = value

        # TODO: output vector images as EPS. overleaf makes svg as rastor (?)

        if motorcycle == 1 & car == 1 & bus == 1 & truck == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 2) | (dataframe["YOLO_id"] == 3) | (dataframe["YOLO_id"] == 5) | (dataframe["YOLO_id"] == 7)]  # noqa: E501
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
        info[key] = ((len(vehicle_ids)/time_[-1]) * 60)
        time_dict = data[key]
        for key_, value in time_dict.items():
            time_cross.append(value)

        if not time_cross:
            info.popitem()
            continue

        time_avg.append(mean(time_cross))

        continents.append(get_single_value_for_key(df_mapping['videos'], df_mapping['continent'], key))  # noqa: E501
        gdp.append(int(get_single_value_for_key(df_mapping['videos'], df_mapping['gdp_per_capita'], key)))  # noqa: E501
        city_.append(get_single_value_for_key(df_mapping['videos'], df_mapping['city'], key))  # noqa: E501
        conditions.append(int(get_single_value_for_key(df_mapping['videos'], df_mapping['time_of_day'], key)))  # noqa: E501

    plot_scatter_diag(x=time_avg, y=info, size=gdp, color=continents, symbol=conditions,
                      city=city_,
                      plot_name=save_as,
                      x_label="Average crossing time (in seconds)",
                      y_label="Number of vehicle detected (normalised)",  # noqa: E501
                      legend_x=0.887, legend_y=0.986)


# On an average how many times a person who is crossing a road will hesitate to do it.   # noqa: E501
def plot_time_to_start_crossing(dfs, person_id=0):
    time_dict, sd_dict = {}, {}
    for location, df in dfs.items():
        data = {}
        crossed_ids = df[(df["YOLO_id"] == person_id)]

        # Makes group based on Unique ID
        crossed_ids_grouped = crossed_ids.groupby("Unique Id")
        condition = int(get_single_value_for_key(df_mapping['videos'], df_mapping['time_of_day'], location))   # noqa: E501
        city = get_single_value_for_key(df_mapping['videos'], df_mapping['city'], location)  # noqa: E501

        # Initialize dictionaries to track sum and sum of squares for each city_condition combination   # noqa: E501
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
                    if (x_values[i] - margin <= x_values[i+1] <= x_values[i] + margin):   # noqa: E501
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            flag = 1
                    elif flag == 1:
                        data[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

                else:
                    if (x_values[i] - margin >= x_values[i+1] >= x_values[i] + margin):   # noqa: E501
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
        sum_squares[city_condition_key] = sum(value**2 for value in data.values())   # noqa: E501

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
            new_sd = ((old_sd ** 2 * n_old + new_sum_squares) / (n_old + n_new) - new_mean ** 2) ** 0.5   # noqa: E501

            time_dict[city_condition_key] = new_mean
            sd_dict[city_condition_key] = new_sd
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
    sorted_day_values = dict(sorted(day_values.items(), key=lambda item: item[1]))   # noqa: E501
    sorted_night_values = dict(sorted(night_values.items(), key=lambda item: item[1]))   # noqa: E501
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
            title="Average time taken by the pedestrian to start crossing the road (in seconds)",   # noqa: E501
            tickvals=[-val for val in range(1, max_value + 1)] + [val for val in range(1, max_value + 1)],   # noqa: E501
            ticktext=[abs(val) for val in range(1, max_value + 1)] + [val for val in range(1, max_value + 1)]   # noqa: E501
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
            text=f"Mean:{mean_}; SD:{sd_value}",
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
            text=f"Mean:{mean_}; SD:{sd_value}",
            font=dict(color='black'),
            showarrow=False,
            xanchor='center',
            yanchor='middle'
        )

    # Plot the figure
    fig.show()

    save_plotly_figure(fig, "time_to_start_cross.html", "time_to_start_cross.png", "time_to_start_cross.svg")  # noqa: E501


def plot_no_of_pedestrian_stop(dfs, person_id=0):
    count_dict = {}
    for location, df in dfs.items():
        data = {}
        count = 0
        city, condition = location.split('_')
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
                    if (x_values[i] - margin <= x_values[i+1] <= x_values[i] + margin):   # noqa: E501
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
                    if (x_values[i] - margin >= x_values[i+1] >= x_values[i] + margin):   # noqa: E501
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
        count_dict[location] = count/1000

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
    sorted_day_values = dict(sorted(day_values.items(), key=lambda item: item[1]))   # noqa: E501
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
            title="No of pedestrian in the study (in thousands)",   # noqa: E501
            tickvals=[-val for val in range(1, max_value + 1)] + [val for val in range(1, max_value + 1)],   # noqa: E501
            ticktext=[abs(val) for val in range(1, max_value + 1)] + [val for val in range(1, max_value + 1)]   # noqa: E501
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

    # Plot the figure
    fig.show()

    save_plotly_figure(fig, "no_of_cases_for_cross.html", "no_of_cases_for_cross.png", "no_of_cases_for_cross.svg")  # noqa: E501


def plot_hesitation_vs_traffic_mortality(df_mapping, dfs, person_id=0):
    count_dict = {}
    time_, traffic_mortality, continents, gdp, conditions = [], [], [], [], []
    for location, df in dfs.items():
        city, condition = location.split('_')
        count, pedestrian_count = 0, 0
        crossed_ids = df[(df["YOLO_id"] == person_id)]

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
                            # Check if there's any instance where X-center [i + 1] <= X-center [i]   # noqa:E501
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
                            # Check if there's any instance where X-center [i + 1] >= X-center [i]   # noqa: E501
                            for j in range(i+1, len(x_values) - 10):
                                if x_values[j] >= x_values[j - 10]:
                                    count += 1
                                    break
                                break

        df_ = df_mapping[(df_mapping['Location'] == city) & (df_mapping['Condition'] == int(condition))]   # noqa: E501

        # num_person = count_object(df, 0)
        if pedestrian_count == 0:
            continue
        time_.append(df_['Duration'].values[0])
        continents.append(df_['Continent'].values[0])
        gdp.append(df_['GDP_per_capita'].values[0])
        traffic_mortality.append(df_['traffic_mortality'].values[0])
        conditions.append(int(condition))
        count_dict[f"{city}_{condition}"] = ((((count * 60) * 100) / pedestrian_count) / time_[-1])  # noqa: E501

    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

    fig = px.scatter(x=list(count_dict.values()),
                     y=traffic_mortality,
                     size=gdp,
                     color=continents,
                     symbol=conditions,
                     labels={"color": "Continent"},
                     color_discrete_map=continent_colors)

    # Hide legend for all traces generated by Plotly Express
    for trace in fig.data:
        trace.showlegend = False

    # Adding labels and title
    fig.update_layout(
        xaxis_title="Percentage of people who hesitated while crossing the road (normalised)",  # noqa: E501
        yaxis_title="Traffic mortality rate per 100k person",  # noqa: E501
    )

    for continent, color in continent_colors.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color), name=continent))  # noqa: E501

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)', line=dict(color='black', width=2)), name=description))  # noqa: E501

    # Adding annotations for keys
    annotations = []
    for i, key in enumerate(count_dict.keys()):
        location_name = key.split('_')[0]
        annotations.append(
            dict(
                x=list(count_dict.values())[i],
                y=traffic_mortality[i],
                text=location_name,
                showarrow=False
            )
        )
    # Adjust annotation positions to avoid overlap
    adjusted_annotations = adjust_annotation_positions(annotations)
    fig.update_layout(annotations=adjusted_annotations)

    # set template
    fig.update_layout(template=template)
    fig.update_layout(legend_title_text=" ")
    fig.update_layout(
            legend=dict(
                x=0.887,
                y=0.986,
                traceorder="normal",
            )
        )
    fig.show()
    save_plotly_figure(fig, "hesitation_vs_traffic_mortality.html", "hesitation_vs_traffic_mortality.png", "hesitation_vs_traffic_mortality.svg")  # noqa: E501


def plot_speed_of_crossing_vs_crossing_decision_time(df_mapping, dfs, data, person_id=0):  # noqa: E501
    avg_speed, time_dict = {}, {}
    continents, gdp, conditions, time_ = [], [], [], []

    for city, df in data.items():
        if df == {}:
            continue
        location, condition = city.split('_')
        value = dfs.get(f"{location}_{condition}")

        df_ = df_mapping[(df_mapping['Location'] == location) & (df_mapping['Condition'] == int(condition))]  # noqa: E501
        length = df_['avg_height(cm)'].values[0]
        time_.append(df_['Duration'].values[0])

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

        avg_speed[city] = sum(speed) / len(speed)

    for location, df in dfs.items():
        data = {}
        city, condition = location.split('_')
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
                    if (x_values[i] - margin <= x_values[i+1] <= x_values[i] + margin):   # noqa: E501
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            flag = 1
                    elif flag == 1:
                        data[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

                else:
                    if (x_values[i] - margin >= x_values[i+1] >= x_values[i] + margin):   # noqa: E501
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

        time_dict[location] = (sum(data.values()) / len(data)) / 30

    ordered_values = []
    for key in time_dict:
        city, condition = key.split('_')

        df_ = df_mapping[(df_mapping['Location'] == city) & (df_mapping['Condition'] == int(condition))]
        conditions.append(int(condition))
        continents.append(df_['Continent'].values[0])
        gdp.append(df_['GDP_per_capita'].values[0])

        ordered_values.append((time_dict[key], avg_speed[key]))

    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

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
        xaxis_title="Average time pedestrian takes for crossing decision (in s)",  # noqa: E501
        yaxis_title="Average speed of pedestrian while crossing the road (in m/s)"   # noqa: E501
    )

    for continent, color in continent_colors.items():
        if continent in continents:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color), name=continent))  # noqa: E501

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)', line=dict(color='black', width=2)), name=description))  # noqa: E501

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
    save_plotly_figure(fig, "speed_of_crossing_vs_crossing_decision_time.html",
                       "speed_of_crossing_vs_crossing_decision_time.png",
                       "speed_of_crossing_vs_crossing_decision_time.svg")


def plot_traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping, dfs, data):   # noqa: E501
    var_exist, var_nt_exist, total_per, ratio = {}, {}, {}, {}
    continents, gdp, traffic_mortality, conditions, time_ = [], [], [], [], []

    # For a specific id of a person search for the first and last occurrence of that id and see if the traffic light was present between it or not.  # noqa: E501
    # Only getting those unique_id of the person who crosses the road

    for city, df in data.items():
        location, condition = city.split('_')
        counter_exists, counter_nt_exists = 0, 0
        df_ = df_mapping[(df_mapping['Location'] == location) & (df_mapping['Condition'] == int(condition))]  # noqa: E501
        time_.append(df_['Duration'].values[0])
        value = dfs.get(f"{location}_{condition}")

        for id, time in df.items():
            unique_id_indices = value.index[value['Unique Id'] == id]
            first_occurrence = unique_id_indices[0]
            last_occurrence = unique_id_indices[-1]

            # Check if YOLO_id = 9 exists within the specified index range
            yolo_id_9_exists = any(value.loc[first_occurrence:last_occurrence, 'YOLO_id'] == 9)  # noqa:E501
            yolo_id_9_not_exists = not any(value.loc[first_occurrence:last_occurrence, 'YOLO_id'] == 9)  # noqa:E501

            if yolo_id_9_exists:
                counter_exists += 1
            if yolo_id_9_not_exists:
                counter_nt_exists += 1

        # Normalising the counters
        var_exist[f"{location}_{condition}"] = ((counter_exists * 60) / time_[-1])   # noqa:E501
        var_nt_exist[f"{location}_{condition}"] = ((counter_nt_exists * 60) / time_[-1])  # noqa:E501

        if (counter_exists + counter_nt_exists) == 0:
            var_exist.popitem()
            var_nt_exist.popitem()
            continue
        else:
            total_per[f"{location}_{condition}"] = var_exist[f"{location}_{condition}"] + var_nt_exist[f"{location}_{condition}"]  # noqa:E501
            # Percentage of people crossing the road without traffic light
            ratio[f"{location}_{condition}"] = (var_nt_exist[f"{location}_{condition}"] * 100) / total_per[f"{location}_{condition}"]  # noqa:E501

        continents.append(df_['Continent'].values[0])
        gdp.append(df_['GDP_per_capita'].values[0])
        traffic_mortality.append(df_['traffic_mortality'].values[0])
        conditions.append(int(condition))

    # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

    fig = px.scatter(x=list(ratio.values()),
                     y=traffic_mortality,
                     size=gdp,
                     color=continents,
                     symbol=conditions,
                     labels={"color": "Continent"},
                     color_discrete_map=continent_colors)

    # Hide legend for all traces generated by Plotly Express
    for trace in fig.data:
        trace.showlegend = False

    # Adding labels and title
    fig.update_layout(
        xaxis_title="Percentage of Crossing Event without traffic light (normalised)",  # noqa: E501
        yaxis_title="Traffic mortality rate (per 100k)"  # noqa: E501
    )

    for continent, color in continent_colors.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color), name=continent))  # noqa: E501

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)', line=dict(color='black', width=2)), name=description))  # noqa: E501

    # Adding annotations for keys
    annotations = []
    for i, key in enumerate(ratio.keys()):
        location_name = key.split('_')[0]
        annotations.append(
            dict(
                x=list(ratio.values())[i],
                y=traffic_mortality[i],
                text=location_name,
                showarrow=False
            )
        )
    # Adjust annotation positions to avoid overlap
    adjusted_annotations = adjust_annotation_positions(annotations)
    fig.update_layout(annotations=adjusted_annotations)

    # set template
    fig.update_layout(template=template)
    fig.update_layout(legend_title_text=" ")
    fig.update_layout(
            legend=dict(
                x=0.05,
                y=0.986,
                traceorder="normal",
            )
        )
    fig.show()
    save_plotly_figure(fig, "traffic_mortality_vs_crossing_event_wt_traffic_light.html",   # noqa: E501
                       "traffic_mortality_vs_crossing_event_wt_traffic_light.png",   # noqa: E501
                       "traffic_mortality_vs_crossing_event_wt_traffic_light.svg")   # noqa: E501


# TODO: markers disappear when sub selection is done
def plot_speed_of_crossing_vs_traffic_mortality(df_mapping, dfs, data):
    avg_speed = {}
    continents, gdp, traffic_mortality, conditions, time_ = [], [], [], [], []
    for city, df in data.items():
        if df == {}:
            continue
        location, condition = city.split('_')
        value = dfs.get(f"{location}_{condition}")

        df_ = df_mapping[(df_mapping['Location'] == location) & (df_mapping['Condition'] == int(condition))]  # noqa: E501
        length = df_['avg_height(cm)'].values[0]
        time_.append(df_['Duration'].values[0])
        conditions.append(int(condition))
        traffic_mortality.append(df_['traffic_mortality'].values[0])
        continents.append(df_['Continent'].values[0])
        gdp.append(df_['GDP_per_capita'].values[0])

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

        avg_speed[city] = sum(speed) / len(speed)

    # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

    fig = px.scatter(x=list(avg_speed.values()),
                     y=traffic_mortality,
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
        xaxis_title="Average speed of the pedestrian to cross the road (in m/s)",  # noqa: E501
        yaxis_title="Traffic mortality rate (per 100k)"
    )

    for continent, color in continent_colors.items():
        if continent in continents:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color), name=continent))  # noqa: E501

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)', line=dict(color='black', width=2)), name=description))  # noqa: E501

    # Adding annotations for locations
    annotations = []
    for i, key in enumerate(avg_speed.keys()):
        location_name = key.split('_')[0]  # Extracting location name
        annotations.append(
            dict(
                x=list(avg_speed.values())[i],
                y=traffic_mortality[i],
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
                x=0.05,
                y=0.986,
                traceorder="normal",
            )
        )

    fig.show()
    save_plotly_figure(fig, "speed_of_crossing_vs_traffic_mortality.html",
                       "speed_of_crossing_vs_traffic_mortality.png",
                       "speed_of_crossing_vs_traffic_mortality.svg")


def plot_speed_of_crossing_vs_literacy(df_mapping, dfs, data):
    avg_speed = {}
    continents, gdp, literacy, conditions, time_ = [], [], [], [], []
    for city, df in data.items():
        if df == {}:
            continue
        location, condition = city.split('_')
        value = dfs.get(f"{location}_{condition}")

        df_ = df_mapping[(df_mapping['Location'] == location) & (df_mapping['Condition'] == int(condition))]  # noqa: E501
        length = df_['avg_height(cm)'].values[0]
        time_.append(df_['Duration'].values[0])
        conditions.append(int(condition))
        literacy.append(df_['Literacy_rate'].values[0])
        continents.append(df_['Continent'].values[0])
        gdp.append(df_['GDP_per_capita'].values[0])

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

        avg_speed[city] = sum(speed) / len(speed)

    # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

    fig = px.scatter(x=list(avg_speed.values()),
                     y=literacy,
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
        xaxis_title="Average speed of the pedestrian to cross the road (in m/s)",  # noqa: E501
        yaxis_title="Literacy rate in the country (in percentage)"
    )

    for continent, color in continent_colors.items():
        if continent in continents:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color), name=continent))  # noqa: E501

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)', line=dict(color='black', width=2)), name=description))  # noqa: E501

    # Adding annotations for locations
    annotations = []
    for i, key in enumerate(avg_speed.keys()):
        location_name = key.split('_')[0]  # Extracting location name
        annotations.append(
            dict(
                x=list(avg_speed.values())[i],
                y=literacy[i],
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
                x=0.887,
                y=0.014,
                traceorder="normal",
            )
        )

    fig.show()
    save_plotly_figure(fig, "speed_of_crossing_vs_literacy.html", "speed_of_crossing_vs_literacy.png", "speed_of_crossing_vs_literacy.svg")  # noqa: E501


def plot_traffic_safety_vs_traffic_mortality(df_mapping, dfs):
    info = {}
    traffic_mortality, continents, gdp, conditions, time_ = [], [], [], [], []
    for key, value in dfs.items():
        location, condition = key.split('_')
        dataframe = value

        df_ = df_mapping[(df_mapping['Location'] == location) & (df_mapping['Condition'] == int(condition))]  # noqa: E501
        time_.append(df_['Duration'].values[0])

        instrument = dataframe[(dataframe["YOLO_id"] == 9) | (dataframe["YOLO_id"] == 11)]  # noqa: E501

        instrument_ids = instrument["Unique Id"].unique()

        if instrument_ids is None:
            continue

        info[key] = ((len(instrument_ids)/time_[-1]) * 60)
        continents.append(df_['Continent'].values[0])
        gdp.append(df_['GDP_per_capita'].values[0])
        traffic_mortality.append(df_['traffic_mortality'].values[0])
        conditions.append(int(condition))

        # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

    fig = px.scatter(x=list(info.values()),
                     y=traffic_mortality,
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
        xaxis_title="Number of traffic instruments detected (normalised)",
        yaxis_title="Traffic mortality rate per 100k person"
    )

    for continent, color in continent_colors.items():
        if continent in continents:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color), name=continent))  # noqa: E501

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)', line=dict(color='black', width=2)), name=description))  # noqa: E501

    # Adding annotations for locations
    annotations = []
    for i, key in enumerate(info.keys()):
        location_name = key.split('_')[0]  # Extracting location name
        annotations.append(
            dict(
                x=list(info.values())[i],
                y=traffic_mortality[i],
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
                x=0.887,
                y=0.05,
                traceorder="normal",
            )
        )

    fig.show()

    save_plotly_figure(fig, "traffic_safety_vs_traffic_mortality.html", "traffic_safety_vs_traffic_mortality.png", "traffic_safety_vs_traffic_mortality.svg")  # noqa: E501


def plot_traffic_safety_vs_literacy(df_mapping, dfs):
    info = {}
    literacy, continents, gdp, conditions, time_ = [], [], [], [], []
    for key, value in dfs.items():
        location, condition = key.split('_')
        dataframe = value

        df_ = df_mapping[(df_mapping['Location'] == location) & (df_mapping['Condition'] == int(condition))]  # noqa: E501
        time_.append(df_['Duration'].values[0])

        instrument = dataframe[(dataframe["YOLO_id"] == 9) | (dataframe["YOLO_id"] == 11)]  # noqa: E501

        instrument_ids = instrument["Unique Id"].unique()

        if instrument_ids is None:
            continue

        info[key] = ((len(instrument_ids)/time_[-1]) * 60)
        continents.append(df_['Continent'].values[0])
        gdp.append(df_['GDP_per_capita'].values[0])
        literacy.append(df_['Literacy_rate'].values[0])
        conditions.append(int(condition))

        # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

    fig = px.scatter(x=list(info.values()),
                     y=literacy,
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
        xaxis_title="Number of traffic instruments detected (normalised)",
        yaxis_title="Literacy rate in the country (in percentage)"
    )

    for continent, color in continent_colors.items():
        if continent in continents:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color), name=continent))  # noqa: E501

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)', line=dict(color='black', width=2)), name=description))  # noqa: E501

    # Adding annotations for locations
    annotations = []
    for i, key in enumerate(info.keys()):
        location_name = key.split('_')[0]  # Extracting location name
        annotations.append(
            dict(
                x=list(info.values())[i],
                y=literacy[i],
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
                x=0.887,
                y=0.05,
                traceorder="normal",
            )
        )
    fig.show()

    save_plotly_figure(fig, "traffic_safety_vs_literacy.html", "traffic_safety_vs_literacy.png", "traffic_safety_vs_literacy.svg")  # noqa: E501


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
        data[key] = time_to_cross(dfs[key], pedestrian_crossing_count[key]["ids"])  # noqa: E501

    # Data is dictionary in the form {City_condition : Values}. Values itself is another dictionary which is {Unique Id of person : Avg time to cross the road} # noqa: E501
    # dfs is a dictionary in the form {City_condition : CSV file}
    # df_mapping is the csv file

    # plot_cell_phone_vs_traffic_mortality(df_mapping, dfs)
    # plot_vehicle_vs_cross_time(df_mapping, dfs, data, motorcycle=1, car=1, bus=1, truck=1)  # noqa: E501
    # plot_traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping, dfs, data)  # noqa: E501
    # plot_hesitation_vs_traffic_mortality(df_mapping, dfs)
    # plot_hesitation_vs_literacy(df_mapping, dfs)

    # plot_speed_of_crossing_vs_traffic_mortality(df_mapping, dfs, data)
    # plot_speed_of_crossing_vs_literacy(df_mapping, dfs, data)

    # plot_traffic_safety_vs_traffic_mortality(df_mapping, dfs)
    # plot_traffic_safety_vs_literacy(df_mapping, dfs)

    plot_time_to_start_crossing(dfs)
    # plot_no_of_pedestrian_stop(dfs)
    # plot_speed_of_crossing_vs_crossing_decision_time(df_mapping, dfs, data)
