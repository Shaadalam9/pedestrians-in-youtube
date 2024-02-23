import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import linregress
import plotly.express as px
from statistics import mean

# List of things that YOLO can detect:
# YOLO_id = {
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
#     9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
#     16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
#     25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
#     33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
#     40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
#     49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',57: 'couch',
#     58: 'potted plant', 59: 'bed',60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
#     66: 'keyboard',67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
#     74: 'clock',75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
# }


def read_csv_files(folder_path):
    dfs = {}
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            # Store the DataFrame in the dictionary with the filename as the key
            filename = os.path.splitext(file)[0]  # Get the filename without extension
            dfs[filename] = df
    return dfs

def pedestrian_crossing(dataframe, min_x, max_x, person_id):

    crossed_ids = dataframe[(dataframe["YOLO_id"] == person_id)]  # sort only person in the dataframe
    crossed_ids_grouped = crossed_ids.groupby("Unique Id")  # Makes group based on Unique ID 
    # for group_name, group_data in crossed_ids_grouped:   
    #     print(f"Group Name: {group_name}")
    #     print(group_data)
    #     print("\n")

    # Filter the unique ID based on passing the pixel value (ranging in x direction from 0 to 1) 
    filtered_crossed_ids = crossed_ids_grouped.filter(
        lambda x: (x["X-center"] <= min_x).any() and (x["X-center"] >= max_x).any()
    )
    
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


    values = []
    keys = []
    for key, sub_dict in data.items():
        for val in sub_dict.values():
            values.append(val)
            keys.append(key)

    # Create histogram figure
    fig = go.Figure()

    # Add histogram traces for each key
    for key in set(keys):
        key_values = [val for val, k in zip(values, keys) if k == key]
        fig.add_trace(go.Histogram(x=key_values, name=key))

    # Update layout
    fig.update_layout(
        title="Histogram of Values from DICT",
        xaxis_title="Values",
        yaxis_title="Frequency",
        bargap=0.2,  # Gap between bars
        barmode='overlay'  # Overlay histograms
    )
    fig.show()
    
def adjust_annotation_positions(annotations):
    adjusted_annotations = []
    for i, ann in enumerate(annotations):
        adjusted_ann = ann.copy()
        # Adjust x and y coordinates to avoid overlap
        for other_ann in adjusted_annotations:
            if (abs(ann['x'] - other_ann['x']) < 2) and (abs(ann['y'] - other_ann['y']) < 1):
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
    filtered_death = [d for i, d in enumerate(death) if info[list(info.keys())[i]] != 0]
    filtered_continents = [c for i, c in enumerate(continents) if info[list(info.keys())[i]] != 0]
    filtered_gdp = [c for i, c in enumerate(gdp) if info[list(info.keys())[i]] != 0] 

    fig = px.scatter(x=filtered_death, y=list(filtered_info.values()), size=filtered_gdp, color=filtered_continents)

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

    fig.show()

def plot_vehicle_vs_cross_time(df_mapping, dfs, data):
    info, time_dict, time_avg, continents, gdp = {}, {}, [], [], []
    for key, value in dfs.items():
        dataframe = value
        vehicle_ids = dataframe[(dataframe["YOLO_id"] == 2) | (dataframe["YOLO_id"] == 3) | (dataframe["YOLO_id"] == 5) | (dataframe["YOLO_id"] == 7)]
        vehicle_ids = vehicle_ids["Unique Id"].unique()
        info[key] = len(vehicle_ids) # contains {location : no of vehicle detected}
        

        time_dict = data[key]
        time = []
        for key_, value in time_dict.items():
            time.append(value)
        time_avg.append(mean(time))

        df = df_mapping[df_mapping['Location'] == key]
        
        continents.append(df['Continent'].values[0])
        gdp.append(df['GDP_per_capita'].values[0])
    
    
    fig = px.scatter(x=time_avg, y=list(info.values()), size=gdp, color=continents)

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

    fig.update_layout(annotations=adjusted_annotations)

    fig.show()

def plot_hesitation():

    pass

def plot_death_vs_crossing_event_wt_traffic(df_mapping, dfs, data, ids):
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


        # For a specefic id of a person search for the first and last occurance of that id and see if the traffic light was present between it or not.

        
def plot_traffic_safety_vs_death():

    pass


data_folder = "data"
dfs = read_csv_files(data_folder)
# print(dfs)
pedestrian_crossing_count, data = {}, {}

for key, value in dfs.items():
    count, ids = pedestrian_crossing(dfs[key], 0.45, 0.55, 0)
    pedestrian_crossing_count[key] = {"count": count, "ids": ids}
    data[key] = time_to_cross(dfs[key], pedestrian_crossing_count[key]["ids"])

# Data is dictionary in the form {City : Values}. Values itself is another dictionary which is {Unique Id of person : Avg time to cross the road}

df_mapping = pd.read_csv("mapping.csv")

plot_cell_phone_vs_death(df_mapping, dfs)
plot_vehicle_vs_cross_time(df_mapping,dfs,data)
# plot_hesitation()
# plot_death_vs_crossing_event_wt_traffic(df_mapping, dfs, data, ids)
