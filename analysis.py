import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import linregress

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


def plot_displot(data):
    final_data = []

    for key, sub_dict in data.items():
        values = list(sub_dict.values())  # Simplified way to get values
        final_data.extend(values)  # Extend instead of append to avoid nested lists

    # Create a DataFrame with a column for the data and a column for the corresponding key
    df = pd.DataFrame({'Data': final_data, 'Key': [k for k in data for _ in range(len(data[k]))]})
    
    plt.figure(figsize=(10, 6))
    with sns.plotting_context(rc={"legend.fontsize": 10}):  # Adjust legend font size
        sns.displot(data=df, x='Data', hue='Key', kind='kde', multiple='stack', legend=True)
    
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.title('Distribution Plot of Final Data')
    plt.legend(title='Key')  # Provide a title for the legend
    plt.show()


def plot_histogram(data):
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
    
def plot_vehicles_vs_GDP():
    pass

def cell_phone_vs_GDP():
    pass




def plot_GDP_vs_time_to_cross(mapping, data):
    gdp, mean_time = [], []
    for key, value in data.items():
        df = mapping[mapping['Location'] == key]
        gdp.append(df['GDP_per_capita'].iloc[0])
        dummy = []
        for val in value.values():
            dummy.append(val)
        mean_time.append(np.mean(dummy))

    plt.scatter(mean_time, gdp, label='Data')

    # Add annotations (keys)
    for i, txt in enumerate(data.keys()):
        plt.annotate(txt, (mean_time[i], gdp[i]), xytext=(5, 5), textcoords='offset points')

    # Fit a line using L1 (Least Absolute Deviations) regression
    slope, intercept, _, _, _ = linregress(mean_time, gdp)
    plt.plot(mean_time, slope * np.array(mean_time) + intercept, color='green',  label='L1 Regression')

    # Fit a line using least squares regression
    # m, b = np.polyfit(mean_time, gdp, 1)
    # plt.plot(mean_time, m * np.array(mean_time) + b, color='red',linestyle=':', label='Least Squares Regression')

    plt.xlabel('Mean Time to Cross')
    plt.ylabel('GDP per Capita')
    plt.title('GDP per Capita vs. Mean Time to Cross')
    plt.legend()
    plt.show()



data_folder = "data"
dfs = read_csv_files(data_folder)
pedestrian_crossing_count, data = {}, {}

for key, value in dfs.items():
    count, ids = pedestrian_crossing(dfs[key], 0.45, 0.55, 0)
    pedestrian_crossing_count[key] = {"count": count, "ids": ids}
    data[key] = time_to_cross(dfs[key], pedestrian_crossing_count[key]["ids"])

# plot_displot(data)
# plot_histogram(data)

df_mapping = pd.read_csv("Mapping.csv")
plot_GDP_vs_time_to_cross(df_mapping,data)