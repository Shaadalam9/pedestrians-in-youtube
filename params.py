from datetime import datetime

y_tube_link = "https://www.youtube.com/watch?v=q21Kj-pxJW4"
resolution = "360p"
output_path = "video"
trim_start = 30 #seconds
trim_end = 31
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