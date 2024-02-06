import pandas as pd
import matplotlib.pyplot as pyplot
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
        

df = df_person["Kuala_Lumpur"]
print(df)
# Find the minimum and maximum x and y coordinates of all bounding boxes
xmin = df['X-center'].min() - df['Width'].max()/2
ymin = df['Y-center'].min() - df['Height'].max()/2
xmax = df['X-center'].max() + df['Width'].max()/2
ymax = df['Y-center'].max() + df['Height'].max()/2

# Define road boundaries using the extracted minimum and maximum values
road_polygon = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

# Initialize a counter for crossed persons
crossed_count = 0

# Loop through each person's bounding box
for index, row in df.iterrows():
    # Extract bounding box variables
    x_centre = row['X-center']
    y_centre = row['Y-center']
    height = row['Height']
    width = row['Width']
    unique_id = row['Unique Id']
    
    # Check if the person's bounding box intersects with the road boundary
    if (xmin <= x_centre <= xmax) and (ymin <= y_centre <= ymax):
        print(f"Person with Unique Id {unique_id} is within the road boundaries.")
    else:
        print(f"Person with Unique Id {unique_id} has crossed the road.")
        crossed_count += 1

# Print total number of persons crossed the road
print(f"Total number of persons crossed the road: {crossed_count}")