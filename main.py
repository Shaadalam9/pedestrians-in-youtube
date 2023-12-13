import params as params
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from moviepy.video.io.VideoFileClip import VideoFileClip
import shutil
from PIL import Image 
import os
from datetime import datetime
from helper import youtube_helper

helper = youtube_helper()


result = helper.download_video_with_resolution(params.y_tube_link, 
                                               resolution=params.resolution, output_path=params.output_path)

if result:
    video_file_path, video_title = result
    print(f"Video title: {video_title}")
    print(f"Video saved at: {video_file_path}")
else:
    print("Download failed.")

input_video_path = f"{params.output_path}/{video_title}_{params.resolution}.mp4"
output_video_path = f"{params.output_path}/{video_title}_{params.resolution}_mod.mp4"

start_time = params.trim_start  # seconds
end_time = params.trim_end  # Set to None to keep the rest of the video

if start_time == None and end_time == None:
    print("No trimming required")
else:
    print("Trimming in progress.......")
    helper.trim_video(input_video_path, output_video_path, start_time, end_time)

os.remove(f"{params.output_path}/{video_title}_{params.resolution}.mp4")
print("Deleted the untrimmed video")
os.rename(output_video_path, input_video_path)

model = YOLO(params.model)
print(f"{video_title}_{params.resolution}")

if params.prediction_mode:
    model.predict(source = f"{params.output_path}/{video_title}_{params.resolution}.mp4", save= True, conf= params.confidence, 
              save_txt=True, show=params.render, line_width = params.line_thickness, show_labels= params.show_labels, 
              show_conf = params.show_conf)


if params.tracking_mode:
    cap = cv2.VideoCapture(input_video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Output paths for frames, txt files, and final video
    frames_output_path = "runs/detect/frames"  
    annotated_frame_output_path = "runs/detect/annotated_frames" 
    txt_output_path = "runs/detect/labels"
    final_video_output_path = "runs/detect/final_video.mp4"  

    # Create directories if they don't exist
    os.makedirs(frames_output_path, exist_ok=True)
    os.makedirs(txt_output_path, exist_ok=True)
    os.makedirs(annotated_frame_output_path, exist_ok=True)

    # Initialize a VideoWriter for the final video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_video_writer = cv2.VideoWriter(final_video_output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    if params.display_frame_tracking:
        display_video_output_path = "runs/detect/display_video.mp4"
        display_video_writer = cv2.VideoWriter(display_video_output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))


    # Loop through the video frames
    frame_count = 0  # Variable to track the frame number
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:

            frame_count += 1  # Increment frame count
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, tracker='bytetrack.yaml', persist=True, conf=params.confidence, save=True, save_txt=True, 
                        line_width=params.line_thickness, show_labels=params.show_labels, show_conf=params.show_labels,
                        show=params.render)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Save annotated frame to file
            if params.save_annoted_img:
                frame_filename = os.path.join(annotated_frame_output_path, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, annotated_frame)


            # Save txt file with bounding box information
            text_filename = "runs/detect/predict/labels/image0.txt"
            with open(text_filename, 'r') as text_file:
                data=text_file.read()
            new_txt_file_name = f"runs/detect/labels/label_{frame_count}.txt"
            with open(new_txt_file_name,'w') as new_file:
                new_file.write(data)

            #save the labelled image
            image_filename = "runs/detect/predict/image0.jpg"
            new_img_file_name = f"runs/detect/frames/frame_{frame_count}.jpg"
            shutil.move(image_filename, new_img_file_name)                    
        
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=params.line_thickness*5)

            # Display the annotated frame
            if params.display_frame_tracking:

                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                display_video_writer.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:

            break


    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    final_video_writer.release()

helper.create_video_from_images(frames_output_path,final_video_output_path,30)
shutil.rmtree("runs/detect/predict")
helper.rename_folder("runs/detect", f"runs/{video_title}_{params.resolution}_{params.timestrap}")

#delete the video downloaded
if params.delete_youtube_video:
    os.remove(f"video/{video_title}_{params.resolution}.mp4")