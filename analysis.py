import pandas as pd
import matplotlib.pyplot as plt
import os

# List of things that YOLO can detect:


# YOLO_id = {
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
#     9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
#     16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
#     25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 
#     33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
#     40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
#     49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',57: 'couch', 
#     58: 'potted plant', 59: 'bed',60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
#     67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
#     75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
# }



def delete_1_min():
    

    pass



def plot_histograms(crossed_id_counts_J, crossed_id_counts_K):
    fig, ax = plt.subplots()

    # Plot histogram for crossed_id_counts_J
    ax.hist(crossed_id_counts_J.values(), bins=20, alpha=0.5, label='Jakarta')

    # Plot histogram for crossed_id_counts_K
    ax.hist(crossed_id_counts_K.values(), bins=20, alpha=0.5, label='Kuala Lumpur')

    ax.set_xlabel('Count')
    ax.set_ylabel('Frequency')
    ax.legend()

    plt.show()

data_folder= 'data'

# Dictionary to store DataFrames of each CSV file
dfs, df_person = {},{}

# Loop through each file in the folder
for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        # Read the CSV file into a DataFrame
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)
        # Store the DataFrame in the dictionary with the filename as the key
        filename = os.path.splitext(file)[0]  # Get the filename without extension
        dfs[filename] = df

        # Filter where only person is present
        df_person[filename] = df[dfs[filename]['YOLO_id'] == 0]
        

df = df_person["Jakarta"]

crossed_ids = df[df['X-center'] > 0.5]['Unique Id'].unique()
previously_less_than_05 = df[(df['X-center'] < 0.5) & (df['Unique Id'].isin(crossed_ids))]['Unique Id'].unique()

if len(crossed_ids) > 0:
    print("IDs that have crossed x-center = 0.5:")
    print(previously_less_than_05)

    crossed_id_counts_J = {}
    for crossed_id in previously_less_than_05:
        count = df[df['Unique Id'] == crossed_id].shape[0]
        crossed_id_counts_J[crossed_id] = count
        
    print("Total number of times each crossed ID was present:")
    print(crossed_id_counts_J)



df = df_person["Kuala_Lumpur"]

crossed_ids = df[df['X-center'] > 0.5]['Unique Id'].unique()
previously_less_than_05 = df[(df['X-center'] < 0.5) & (df['Unique Id'].isin(crossed_ids))]['Unique Id'].unique()

if len(crossed_ids) > 0:
    print("IDs that have crossed x-center = 0.5:")
    print(previously_less_than_05)

    crossed_id_counts_K = {}
    for crossed_id in previously_less_than_05:
        count = df[df['Unique Id'] == crossed_id].shape[0]
        crossed_id_counts_K[crossed_id] = count
        
    print("Total number of times each crossed ID was present:")
    print(crossed_id_counts_K)


plot_histograms(crossed_id_counts_J, crossed_id_counts_K)