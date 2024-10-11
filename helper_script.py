import os
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from ultralytics import YOLO
import params as params
from collections import defaultdict
import shutil
import numpy as np
import pandas as pd
from custom_logger import CustomLogger


logger = CustomLogger(__name__)  # use custom logger


class youtube_helper:

    def __init__(self, video_title=None):
        self.model = params.model
        self.resolution = None
        self.video_title = video_title

    def set_video_title(self, title):
        self.video_title = title

    @staticmethod
    def rename_folder(old_name, new_name):
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            logger.error(f"Error: Folder '{old_name}' not found.")
        except FileExistsError:
            logger.error(f"Error: Folder '{new_name}' already exists.")

    def download_video_with_resolution(self, video_id, resolutions=["720p", "480p", "360p"], output_path="."):
        try:
            youtube_url = f'https://www.youtube.com/watch?v={video_id}'
            youtube_object = YouTube(youtube_url, on_progress_callback=on_progress)
            for resolution in resolutions:
                video_streams = youtube_object.streams.filter(res=f"{resolution}").all()
                if video_streams:
                    self.resolution = resolution
                    logger.info(f"Got the video in {resolution}")
                    break

            if not video_streams:
                logger.error(f"No {resolution} resolution available for '{youtube_object.title}'.")
                return None

            selected_stream = video_streams[0]

            video_file_path = f"{output_path}/{video_id}.mp4"
            logger.info("Youtube video download in progress...")
            # Comment the below line to automatically download with video in "video" folder
            selected_stream.download(output_path, filename=f"{video_id}.mp4")

            logger.info(f"Download of '{youtube_object.title}' in {resolution} completed successfully.")
            self.video_title = youtube_object.title
            return video_file_path, video_id, resolution

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

    @staticmethod
    def trim_video(input_path, output_path, start_time, end_time):
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        video_clip.close()

    @staticmethod
    def create_video_from_images(image_folder, output_video_path, frame_rate=30):
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

        if not images:
            logger.error("No JPG images found in the specified folder.")
            return

        images.sort(key=lambda x: int(x.split("frame_")[1].split(".")[0]))

        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        for image in images:
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)

            if frame is not None:
                video.write(frame)
            else:
                logger.error(f"Failed to read frame: {img_path}")

        video.release()
        logger.info(f"Video created successfully at: {output_video_path}")

    @staticmethod
    def merge_txt_to_csv_dynamically(txt_location, output_csv, frame_count):
        # Define the path for the new text file
        new_txt_file_name = os.path.join(txt_location, f"label_{frame_count}.txt")

        # Read data from the new text file
        with open(new_txt_file_name, 'r') as text_file:
            data = text_file.read()

        # Save the data into the new text file
        with open(new_txt_file_name, 'w') as new_file:
            new_file.write(data)

        # Read the newly created text file into a DataFrame
        df = pd.read_csv(new_txt_file_name, delimiter=" ", header=None,
                         names=["YOLO_id", "X-center", "Y-center", "Width", "Height", "Unique Id"])

        # Append the DataFrame to the CSV file
        if not os.path.exists(output_csv):
            df.to_csv(output_csv, index=False, mode='w')  # If the CSV does not exist, create it
        else:
            df.to_csv(output_csv, index=False, mode='a', header=False)  # If it exists, append without header

    def prediction_mode(self):
        model = YOLO(self.model)
        model.predict(source=f"{params.output_path}/{self.video_title}_{self.resolution}.mp4",
                      save=True, conf=params.confidence, save_txt=True,
                      show=params.render, line_width=params.line_thickness,
                      show_labels=params.show_labels, show_conf=params.show_conf)

    def tracking_mode(self, input_video_path, output_video_path):
        model = YOLO(self.model)
        cap = cv2.VideoCapture(input_video_path)

        # Store the track history
        track_history = defaultdict(lambda: [])

    # Output paths for frames, txt files, and final video
        frames_output_path = "runs/detect/frames"
        annotated_frame_output_path = "runs/detect/annotated_frames"
        txt_output_path = "runs/detect/labels"
        final_video_output_path = "runs/detect/final_video.mp4"
        text_filename = "runs/detect/predict/labels/image0.txt"

        # Create directories if they don't exist
        os.makedirs(frames_output_path, exist_ok=True)
        os.makedirs(txt_output_path, exist_ok=True)
        os.makedirs(annotated_frame_output_path, exist_ok=True)

    # Initialize a VideoWriter for the final video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        final_video_writer = cv2.VideoWriter(final_video_output_path,
                                             fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        if params.display_frame_tracking:
            display_video_output_path = "runs/detect/display_video.mp4"
            display_video_writer = cv2.VideoWriter(display_video_output_path,
                                                   fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        # Loop through the video frames
        frame_count = 0  # Variable to track the frame number
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:

                frame_count += 1  # Increment frame count
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, tracker='bytetrack.yaml',
                                      persist=True, conf=params.confidence,
                                      save=True, save_txt=True,
                                      line_width=params.line_thickness,
                                      show_labels=params.show_labels,
                                      show_conf=params.show_labels,
                                      show=params.render)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                if boxes.size(0) == 0:
                    with open(text_filename, 'w') as file:   # noqa: F841
                        pass

                try:
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()

                # Save annotated frame to file
                    if params.save_annoted_img:
                        frame_filename = os.path.join(annotated_frame_output_path, f"frame_{frame_count}.jpg")
                        cv2.imwrite(frame_filename, annotated_frame)

                except Exception:
                    pass

                # Save txt file with bounding box information
                with open(text_filename, 'r') as text_file:
                    data = text_file.read()
                new_txt_file_name = f"runs/detect/labels/label_{frame_count}.txt"
                with open(new_txt_file_name, 'w') as new_file:
                    new_file.write(data)
                youtube_helper.merge_txt_to_csv_dynamically("runs/detect/labels",
                                                            f"runs/detect/{self.video_title}.csv", frame_count)
                os.remove(text_filename)
                if params.delete_labels is True:
                    os.remove(f"runs/detect/labels/label_{frame_count}.txt")

                # save the labelled image
                if params.delete_frames is False:
                    image_filename = "runs/detect/predict/image0.jpg"
                    new_img_file_name = f"runs/detect/frames/frame_{frame_count}.jpg"
                    shutil.move(image_filename, new_img_file_name)

                # Plot the tracks
                try:
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)

                    # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230),
                                      thickness=params.line_thickness*5)

                except Exception:
                    pass

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
