input_csv_file = "mapping.csv"
output_path = "video"
trim_start = 0 #seconds
trim_end = 2

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

# Output paths for frames, txt files, and final video
frames_output_path = "runs/detect/frames"  
annotated_frame_output_path = "runs/detect/annotated_frames" 
txt_output_path = "runs/detect/labels"
final_video_output_path = "runs/detect/final_video.mp4" 
output_merged_csv = "runs/detect/merged_output.csv"