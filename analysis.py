import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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


# delete the first 1 min from the csv files.
def delete_1_min(dataframe, rows=30 * 60):
    return dataframe.iloc[120:]


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
            dfs[filename] = delete_1_min(dfs[filename])
    return dfs


def plot_histograms(crossed_id_counts_J, crossed_id_counts_K):
    fig, ax = plt.subplots()

    # Plot histogram for crossed_id_counts_J
    ax.hist(crossed_id_counts_J.values(), bins=20, alpha=0.5, label="Jakarta")

    # Plot histogram for crossed_id_counts_K
    ax.hist(crossed_id_counts_K.values(), bins=20, alpha=0.5, label="Kuala Lumpur")

    ax.set_xlabel("Count")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.show()


def pedestrian_crossing(dataframe, min_x, max_x, person_id):

    crossed_ids = dataframe[(dataframe["YOLO_id"] == person_id)]
    crossed_ids_grouped = crossed_ids.groupby("Unique Id")
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


def plot_line_graph_ped_crossing(DICT):
    values = []
    keys = []
    for key, sub_dict in DICT.items():
        values.extend(sub_dict.values())
        keys.extend([key] * len(sub_dict))

    # Create a color map
    unique_keys = list(set(keys))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_keys)))
    color_map = {key: colors[i] for i, key in enumerate(unique_keys)}

    # Plot line graph
    plt.figure(figsize=(8, 6))
    for key in unique_keys:
        key_values = [value for value, k in zip(values, keys) if k == key]
        plt.plot(sorted(key_values), label=key, color=color_map[key])

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Line Graph of Values with Color-coded Keys')

    # Add legend
    plt.legend(loc='upper right')

    # Show plot
    plt.show()

def plot_histogram_ped_crossing(DICT):
    # Flatten the dictionaries
    values = []
    keys = []
    for key, sub_dict in DICT.items():
        values.extend(sub_dict.values())
        keys.extend([key] * len(sub_dict))

    # Create a color map
    unique_keys = list(set(keys))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_keys)))
    color_map = {key: colors[i] for i, key in enumerate(unique_keys)}

    # Plot histogram
    plt.figure(figsize=(8, 6))
    for key in unique_keys:
        key_values = [value for value, k in zip(values, keys) if k == key]
        plt.hist(key_values, bins=20, alpha=0.7, color=color_map[key], label=key)

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values with Color-coded Keys')

    # Add legend
    plt.legend(loc='upper right')

    # Show plot
    plt.show()


data_folder = "data"
dfs = read_csv_files(data_folder)
pedestrian_crossing_count, data = {}, {}

for key, value in dfs.items():
    count, ids = pedestrian_crossing(dfs[key], 0.45, 0.55, 0)
    pedestrian_crossing_count[key] = {"count": count, "ids": ids}
    data[key] = time_to_cross(dfs[key], pedestrian_crossing_count[key]["ids"])

# print(data["Jakarta"])
plot_histogram_ped_crossing(data)
plot_line_graph_ped_crossing(data)