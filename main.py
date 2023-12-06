# compare one or multiple metrics across countries (cross-cultural effects)
#   1. presence of eye contact/hand gestures
#   2. amount of time before making the crossing decision
#   3. speed of crossing
#   4. hesitation (tbd)

import params as params
from pytube import YouTube
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def download_video_with_resolution(youtube_url, resolution="720p", output_path="."):
    try:
        youtube_object = YouTube(youtube_url)    # Create a YouTube object

        video_streams = youtube_object.streams.filter(res=f"{resolution}").all()


        if not video_streams:
            print(f"No {resolution} resolution available for '{youtube_object.title}'.")
            return None

        selected_stream = video_streams[0]

        video_file_path = f"{output_path}/{youtube_object.title}_{resolution}.mp4"
        selected_stream.download(output_path, filename=f"{youtube_object.title}_{resolution}.mp4")  #Download command 

        print(f"Download of '{youtube_object.title}' in {resolution} completed successfully.")
        
        return video_file_path, youtube_object.title
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#Trimming of video for removing the heading
def trim_video(input_path, output_path, start_time, end_time):
    video_clip = VideoFileClip(input_path).subclip(start_time, end_time)
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    video_clip.close()


result = download_video_with_resolution(params.y_tube_link, resolution=params.resolution, output_path=params.output_path)

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
    trim_video(input_video_path, output_video_path, start_time, end_time)

os.remove(f"{params.output_path}/{video_title}_{params.resolution}.mp4")
print("Deleted the untrimmed video")
os.rename(output_video_path, input_video_path)

model = YOLO(params.model)
print(f"{video_title}_{params.resolution}")

if params.prediction_mode:
    model.predict(source = f"{params.output_path}/{video_title}_{params.resolution}.mp4", save= True, conf= params.confidence, 
              save_txt=True, show=params.render, line_thickness = params.line_thickness, hide_labels= params.hide_labels, hide_conf = params.hide_conf)


if params.tracking_mode:
    cap = cv2.VideoCapture(input_video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # print(success, frame)

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, save=True, save_txt=True, line_width = params.line_thickness,
            show_labels= params.hide_labels, show_conf = params.hide_conf)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()