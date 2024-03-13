import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from statistics import mean
import common
from custom_logger import CustomLogger
from logmod import logs


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


# Plotting Functions
def plot_cell_phone_vs_traffic_mortality(df_mapping, dfs):
    info = {}
    time_, traffic_mortality, continents, gdp, conditions = [], [], [], [], []
    for key, value in dfs.items():
        location, condition = key.split('_')
        dataframe = value

        df = df_mapping[(df_mapping['Location'] == location) & (df_mapping['Condition'] == int(condition))]  # noqa: E501

        mobile_ids = dataframe[dataframe["YOLO_id"] == 67]
        mobile_ids = mobile_ids["Unique Id"].unique()
        time_.append(df['Duration'].values[0])
        num_person = count_object(value, 0)

        # Saving the number of unique mobile discovered in the video
        info[f"{location}_{condition}"] = (((len(mobile_ids) * 60) / time_[-1]) / num_person)  # noqa: E501
        conditions.append(int(condition))

        traffic_mortality.append(df['traffic_mortality'].values[0])
        continents.append(df['Continent'].values[0])
        gdp.append(df['GDP_per_capita'].values[0])

    # Filter out values where info[key] == 0
    filtered_info = {k: v for k, v in info.items() if v != 0}
    filtered_traffic_mortality = [d for i, d in enumerate(traffic_mortality) if info[list(info.keys())[i]] != 0]  # noqa: E501
    filtered_continents = [c for i, c in enumerate(continents) if info[list(info.keys())[i]] != 0]  # noqa: E501
    filtered_gdp = [c for i, c in enumerate(gdp) if info[list(info.keys())[i]] != 0]   # noqa: E501
    filtered_conditions = [c for i, c in enumerate(conditions) if info[list(info.keys())[i]] != 0]   # noqa: E501

    # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

    fig = px.scatter(x=filtered_traffic_mortality,
                     y=list(filtered_info.values()),
                     size=filtered_gdp,
                     color=filtered_continents,
                     symbol=filtered_conditions,  # Use conditions for symbols
                     labels={"color": "Continent"},  # Rename color legend
                     color_discrete_map=continent_colors)

    # Hide legend for all traces generated by Plotly Express
    for trace in fig.data:
        trace.showlegend = False

    # Adding labels and title
    fig.update_layout(
        xaxis_title="Traffic mortality rate per 100k person",  # noqa: E501
        yaxis_title="Number of Mobile detected in the video (normalised)"
    )

    for continent, color in continent_colors.items():
        if continent in filtered_continents:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color), name=continent))  # noqa: E501

    # Adding manual legend for symbols
    symbols_legend = {'triangle-up': 'Night', 'circle': 'Day'}
    for symbol, description in symbols_legend.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(symbol=symbol, color='rgba(0,0,0,0)', line=dict(color='black', width=2)), name=description))  # noqa: E501

    # Adding annotations for locations
    annotations = []
    for i, key in enumerate(filtered_info.keys()):
        location_name = key.split('_')[0]  # Extracting location name
        annotations.append(
            dict(
                x=filtered_traffic_mortality[i],
                y=list(filtered_info.values())[i],
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
                y=0.986,
                traceorder="normal",
            )
        )

    fig.show()
    save_plotly_figure(fig, "cell_phone_vs_traffic_mortality.html", "cell_phone_vs_traffic_mortality.png", "cell_phone_vs_traffic mortality.svg")  # noqa: E501


# TODO: check if there is a csv with avg vehicle ownership/usage on the city/country level   # noqa: E501
def plot_vehicle_vs_cross_time(df_mapping, dfs, data, motorcycle=0, car=0, bus=0, truck=0):    # noqa: E501
    info, time_dict = {}, {}
    time_avg, continents, gdp, conditions, time_ = [], [], [], [], []

    for key, value in dfs.items():
        location, condition = key.split('_')
        df_ = df_mapping[(df_mapping['Location'] == location) & (df_mapping['Condition'] == int(condition))]  # noqa: E501
        time_.append(df_['Duration'].values[0])
        time_cross = []
        dataframe = value

        # TODO: output vector images as EPS. overleaf makes svg as rastor (?)

        if motorcycle == 1 & car == 1 & bus == 1 & truck == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 2) | (dataframe["YOLO_id"] == 3) | (dataframe["YOLO_id"] == 5) | (dataframe["YOLO_id"] == 7)]  # noqa: E501
            html_file = "all_vehicle_vs_cross_time.html"
            png_file = "all_vehicle_vs_cross_time.png"
            svg_file = "all_vehicle_vs_cross_time.svg"

        elif motorcycle == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 2)]
            html_file = "motorcycle_vs_cross_time.html"
            png_file = "motorcycle_vs_cross_time.png"
            svg_file = "motorcycle_vs_cross_time.svg"

        elif car == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 3)]
            html_file = "car_vs_cross_time.html"
            png_file = "car_vs_cross_time.png"
            svg_file = "car_vs_cross_time.svg"

        elif bus == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 5)]
            html_file = "bus_vs_cross_time.html"
            png_file = "bus_vs_cross_time.png"
            svg_file = "bus_vs_cross_time.svg"

        elif truck == 1:
            vehicle_ids = dataframe[(dataframe["YOLO_id"] == 7)]
            html_file = "truck_vs_cross_time.html"
            png_file = "truck_vs_cross_time.png"
            svg_file = "truck_vs_cross_time.svg"

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

        continents.append(df_['Continent'].values[0])
        gdp.append(df_['GDP_per_capita'].values[0])
        conditions.append(int(condition))

    # Hard coded colors for continents
    continent_colors = {'Asia': 'blue', 'Europe': 'green', 'Africa': 'red', 'North America': 'orange', 'South America': 'purple', 'Australia': 'brown'}  # noqa: E501

    fig = px.scatter(x=time_avg,
                     y=list(info.values()),
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
        xaxis_title="Average crossing time (in seconds)",
        yaxis_title="Number of vehicle detected (normalised)",
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
                x=time_avg[i],
                y=list(info.values())[i],
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
                y=0.986,
                traceorder="normal",
            )
        )

    fig.show()

    save_plotly_figure(fig, html_file, png_file, svg_file)


# On an average how many times a person who is crossing a road will hesitate to do it.   # noqa: E501
def plot_time_to_start_crossing(df_mapping, dfs, person_id=0):
    time_dict = {}
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


# TODO: Complete this
def plot_hesitation_vs_literacy(df_mapping, dfs, person_id=0):
    pass


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

    df_mapping = pd.read_csv("mapping.csv")

    # plot_cell_phone_vs_traffic_mortality(df_mapping, dfs)
    # plot_vehicle_vs_cross_time(df_mapping, dfs, data, motorcycle=1, car=1, bus=1, truck=1)  # noqa: E501
    # plot_traffic_mortality_vs_crossing_event_wt_traffic_light(df_mapping, dfs, data)  # noqa: E501
    # plot_hesitation_vs_traffic_mortality(df_mapping, dfs)
    # plot_hesitation_vs_literacy(df_mapping, dfs)

    # plot_speed_of_crossing_vs_traffic_mortality(df_mapping, dfs, data)
    # plot_speed_of_crossing_vs_literacy(df_mapping, dfs, data)

    # plot_traffic_safety_vs_traffic_mortality(df_mapping, dfs)
    # plot_traffic_safety_vs_literacy(df_mapping, dfs)

    # plot_time_to_start_crossing(df_mapping, dfs)
    plot_no_of_pedestrian_stop(dfs)
