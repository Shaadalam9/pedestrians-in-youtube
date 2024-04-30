input_csv_file = "mapping.csv"
output_path = "video"

prediction_mode = False
tracking_mode = True
display_frame_tracking = False
save_annoted_img = False
delete_frames = True
delete_youtube_video = True
need_annotated_video = True

model = "yolov8x.pt"
confidence = 0.7
render = True
line_thickness = 1
show_conf = True
show_labels = True

# Output paths for frames, txt files, and final video
frames_output_path = "runs/detect/frames"
annotated_frame_output_path = "runs/detect/annotated_frames"
txt_output_path = "runs/detect/labels"
final_video_output_path = "runs/detect/final_video.mp4"
output_merged_csv = "runs/detect/merged_output.csv"
