from datetime import datetime
from helper import youtube_helper
import pandas as pd

df = pd.read_csv("mapping.csv")
y_tube_link = "https://www.youtube.com/watch?v=CftLBPI1Ga4"
output_path = "video"
trim_start = 0 #seconds
trim_end = 2
timestrap = datetime.now()

prediction_mode = False
tracking_mode = True
display_frame_tracking = True
save_annoted_img = True
delete_youtube_video = True

model = "yolov8x.pt" 
confidence = 0.4
render = True
line_thickness = 1
show_conf = False
show_labels = False