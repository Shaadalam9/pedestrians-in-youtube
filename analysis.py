import pandas as pd
import os
import plotly.express as px
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
            # Store the DataFrame in the dictionary with the filename as the
            # key
            # Get the filename without extension
            filename = os.path.splitext(file)[0]
            dfs[filename] = df
    return dfs


def pedestrian_crossing(dataframe, min_x, max_x, person_id):
    # sort only person in the dataframe
    crossed_ids = dataframe[(dataframe["YOLO_id"] == person_id)]
    # Makes group based on Unique ID
    crossed_ids_grouped = crossed_ids.groupby("Unique Id")
    # for group_name, group_data in crossed_ids_grouped:
    #     print(f"Group Name: {group_name}")
    #     print(group_data)
    #     print("\n")

    # Filter the unique ID based on passing the pixel value (ranging in x
    # direction from 0 to 1)
    filtered_crossed_ids = crossed_ids_grouped.filter(
        lambda x: (x["X-center"] <= min_x).any() and (x["X-center"] >= max_x).any())  # noqa: E501
    crossed_ids = filtered_crossed_ids["Unique Id"].unique()
    return len(crossed_ids), crossed_ids


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
            if (abs(ann['x'] - other_ann['x']) < 2) and (abs(ann['y'] - other_ann['y']) < 1):  # noqa: E501
                adjusted_ann['y'] += 0.2
        adjusted_annotations.append(adjusted_ann)
    return adjusted_annotations


# Plotting Functions


def plot_cell_phone_vs_death(df_mapping, data):
    info, death, continents, gdp = {}, [], [], []
    for key, value in data.items():
        dataframe = value
        mobile_ids = dataframe[dataframe["YOLO_id"] == 67]
        mobile_ids = mobile_ids["Unique Id"].unique()
        info[key] = len(mobile_ids)

        df = df_mapping[df_mapping['Location'] == key]
        death_value = df['death(per_100k)'].values
        death.append(death_value[0])
        continents.append(df['Continent'].values[0])
        gdp.append(df['GDP_per_capita'].values[0])

    # Filter out values where info[key] == 0
    filtered_info = {k: v for k, v in info.items() if v != 0}
    filtered_death = [d for i, d in enumerate(death) if info[list(info.keys())[i]] != 0]  # noqa: E501
    filtered_continents = [c for i, c in enumerate(continents) if info[list(info.keys())[i]] != 0]  # noqa: E501
    filtered_gdp = [c for i, c in enumerate(gdp) if info[list(info.keys())[i]] != 0]   # noqa: E501

    fig = px.scatter(x=filtered_death,
                     y=list(filtered_info.values()),
                     size=filtered_gdp,
                     color=filtered_continents)

    # Adding labels and title
    fig.update_layout(
        xaxis_title="Death per 100k",
        yaxis_title="Number of Mobile",
        title="Cell Phone Usage vs Death Rate",
        showlegend=True  # Show legend for continent colors
    )

    # Adding annotations for keys
    annotations = []
    for i, key in enumerate(filtered_info.keys()):
        annotations.append(
            dict(
                x=filtered_death[i],
                y=list(filtered_info.values())[i],
                text=key,
                showarrow=False
            )
        )
    # Adjust annotation positions to avoid overlap
    adjusted_annotations = adjust_annotation_positions(annotations)
    fig.update_layout(annotations=adjusted_annotations)
    # set template
    fig.update_layout(template=template)
    fig.show()


def plot_vehicle_vs_cross_time(df_mapping, dfs, data):
    info, time_dict, time_avg, continents, gdp = {}, {}, [], [], []
    for key, value in dfs.items():
        dataframe = value
        vehicle_ids = dataframe[(dataframe["YOLO_id"] == 2) | (dataframe["YOLO_id"] == 3) | (dataframe["YOLO_id"] == 5) | (dataframe["YOLO_id"] == 7)]  # noqa: E501
        vehicle_ids = vehicle_ids["Unique Id"].unique()
        # contains {location : no of vehicle detected}
        info[key] = len(vehicle_ids)

        time_dict = data[key]
        time = []
        for key_, value in time_dict.items():
            time.append(value)
        time_avg.append(mean(time))

        df = df_mapping[df_mapping['Location'] == key]

        continents.append(df['Continent'].values[0])
        gdp.append(df['GDP_per_capita'].values[0])

    fig = px.scatter(x=time_avg,
                     y=list(info.values()),
                     size=gdp,
                     color=continents)

    # Adding labels and title
    fig.update_layout(
        xaxis_title="Avg. time to cross",
        yaxis_title="Number of vehicle detected",
        title="Time to cross vs vehicles",
        showlegend=True  # Show legend for continent colors
    )

    # Adding annotations for keys
    annotations = []
    for i, key in enumerate(info.keys()):
        annotations.append(
            dict(
                x=time_avg[i],
                y=list(info.values())[i],
                text=key,
                showarrow=False
            )
        )
    # Adjust annotation positions to avoid overlap
    adjusted_annotations = adjust_annotation_positions(annotations)

    fig.update_layout(template=template, annotations=adjusted_annotations)
    # set template
    fig.update_layout(template=template)
    fig.show()


def plot_hesitation():
    pass


# For a specific id of a person search for the first and last occurrence of that id and see if the traffic light was present between it or not.  # noqa: E501
def plot_death_vs_crossing_event_wt_traffic(df_mapping, dfs, data, ids):
    var_exist, var_nt_exist, total_per, ratio = {}, {}, {}, {}
    continents, gdp, death = [], [], []
    for city, df in data.items():
        counter_exists, counter_nt_exists = 0, 0
        df_ = df_mapping[df_mapping['Location'] == city]
        continents.append(df_['Continent'].values[0])
        gdp.append(df_['GDP_per_capita'].values[0])
        death.append(df_['death(per_100k)'].values[0])
        value = dfs.get(city)
        for id, time in df.items():
            unique_id_indices = value.index[value['Unique Id'] == id]
            first_occurrence = unique_id_indices[0]
            last_occurrence = unique_id_indices[-1]
            # Check if YOLO_id = 9 exists within the specified index range
            yolo_id_9_exists = any(value.loc[first_occurrence:last_occurrence, 'YOLO_id'] == 9)
            yolo_id_9_not_exists = not any(
                value.loc[first_occurrence:last_occurrence, 'YOLO_id'] == 9)
            if yolo_id_9_exists:
                counter_exists += 1
            if yolo_id_9_not_exists:
                counter_nt_exists += 1
        var_exist[city] = counter_exists
        var_nt_exist[city] = counter_nt_exists
        total_per[city] = counter_exists+counter_nt_exists
        ratio[city] = (var_nt_exist[city] * 100) / total_per[city]

    fig = px.scatter(x=list(var_nt_exist.values()),
                     y=death,
                     size=gdp,
                     color=continents)

    # Adding labels and title
    fig.update_layout(
        xaxis_title="No of Crossing Event",
        yaxis_title="Death rate (per 100k)",
        title="Crossing of pedestrain without traffic light",
        showlegend=True  # Show legend for continent colors
    )

    # Adding annotations for keys
    annotations = []
    for i, key in enumerate(var_nt_exist.keys()):
        annotations.append(
            dict(
                x=list(var_nt_exist.values())[i],
                y=death[i],
                text=key,
                showarrow=False
            )
        )
    # Adjust annotation positions to avoid overlap
    adjusted_annotations = adjust_annotation_positions(annotations)

    fig.update_layout(annotations=adjusted_annotations)
    # set template
    fig.update_layout(template=template)
    fig.show()

    # Percentage of people crossed without traffic light

    fig = px.scatter(x=list(ratio.values()),
                     y=death,
                     size=gdp,
                     color=continents)

    # Adding labels and title
    fig.update_layout(
        xaxis_title="Percentage of people crossing without traffic light",
        yaxis_title="Death rate (per 100k)",
        title="Crossing of pedestrain without traffic light",
        showlegend=True  # Show legend for continent colors
    )

    # Adding annotations for keys
    annotations = []
    for i, key in enumerate(ratio.keys()):
        annotations.append(
            dict(
                x=list(ratio.values())[i],
                y=death[i],
                text=key,
                showarrow=False
            )
        )
    # Adjust annotation positions to avoid overlap
    adjusted_annotations = adjust_annotation_positions(annotations)

    fig.update_layout(annotations=adjusted_annotations)
    # set template
    fig.update_layout(template=template)
    fig.show()


def plot_traffic_safety_vs_death():
    pass


# Execute analysis
logger.info("Analysis started.")
dfs = read_csv_files(common.get_configs('data'))
pedestrian_crossing_count, data = {}, {}

# Loop over rows of data
for key, value in dfs.items():
    logger.info("Analysing data from {}.", key)
    count, ids = pedestrian_crossing(dfs[key], 0.45, 0.55, 0)
    pedestrian_crossing_count[key] = {"count": count, "ids": ids}
    data[key] = time_to_cross(dfs[key], pedestrian_crossing_count[key]["ids"])


# Data is dictionary in the form {City : Values}. Values itself is another
# dictionary which is {Unique Id of person : Avg time to cross the road}

df_mapping = pd.read_csv("mapping.csv")

# plot_cell_phone_vs_death(df_mapping, dfs)
# plot_vehicle_vs_cross_time(df_mapping, dfs, data)
# plot_hesitation()
plot_death_vs_crossing_event_wt_traffic(df_mapping, dfs, data, ids)
