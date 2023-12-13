import os
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2

class youtube_helper:

    def __init__(self):
        pass

    @staticmethod
    def rename_folder(old_name, new_name):
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            print(f"Error: Folder '{old_name}' not found.")
        except FileExistsError:
            print(f"Error: Folder '{new_name}' already exists.")

    @staticmethod
    def download_video_with_resolution(youtube_url, resolution="720p", output_path="."):
        try:
            youtube_object = YouTube(youtube_url)

            video_streams = youtube_object.streams.filter(res=f"{resolution}").all()

            if not video_streams:
                print(f"No {resolution} resolution available for '{youtube_object.title}'.")
                return None

            selected_stream = video_streams[0]

            video_file_path = f"{output_path}/{youtube_object.title}_{resolution}.mp4"
            print("Youtube video download in progress...")
            selected_stream.download(output_path, filename=f"{youtube_object.title}_{resolution}.mp4")

            print(f"Download of '{youtube_object.title}' in {resolution} completed successfully.")
            return video_file_path, youtube_object.title
        except Exception as e:
            print(f"An error occurred: {e}")
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
            print("No JPG images found in the specified folder.")
            return

        images.sort(key=lambda x: int(x.split("frame_")[1].split(".")[0]))

        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        for image in images:
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)

            if frame is not None:
                video.write(frame)
            else:
                print(f"Failed to read frame: {img_path}")

        video.release()
        print(f"Video created successfully at: {output_video_path}")
