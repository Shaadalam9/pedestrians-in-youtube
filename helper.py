import os
from pytube import YouTube
import pandas as pd
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
    def download_video_with_resolution(youtube_url, resolutions=["2160p", "1440p", "1080p", "720p", "480p"], output_path="."):
        try:
            youtube_object = YouTube(youtube_url)
            for resolution in resolutions:
                video_streams = youtube_object.streams.filter(res=f"{resolution}").all()
                if video_streams:
                    print(f"Got the video in {resolution}")
                    break

            if not video_streams:
                print(f"No {resolution} resolution available for '{youtube_object.title}'.")
                return None

            selected_stream = video_streams[0]

            video_file_path = f"{output_path}/{youtube_object.title}_{resolution}.mp4"
            print("Youtube video download in progress...")
            selected_stream.download(output_path, filename=f"{youtube_object.title}_{resolution}.mp4")

            print(f"Download of '{youtube_object.title}' in {resolution} completed successfully.")
            return video_file_path, youtube_object.title, resolution
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

    # @staticmethod
    # def get_highest_resolution(video):
    #     video_streams = video.streams.filter(file_extension="mp4", progressive=True)
    
    #     # Extract resolutions from video streams
    #     resolutions = [stream.resolution for stream in video_streams if stream.resolution]

    #     # Extract the numeric values from the resolutions
    #     resolutions_numeric = [int(res[:-1]) for res in resolutions]

    #     # Find the maximum resolution
    #     max_resolution = max(resolutions_numeric, default=None)

    #     return f"{max_resolution}p" if max_resolution is not None else None


    
    # def process_and_update_csv(input_file):
    #     df = pd.read_csv(input_file)
    #     youtube_link = df['Youtube link']

    #     for link in youtube_link:
    #         outube_object = YouTube(youtube_link)
            


            

    #     # processed_links = set()

    #     # with open(input_file, mode='r+') as csv_file:
    #     #     reader = csv.reader(csv_file)
    #     #     writer = csv.writer(csv_file)
        
    #     #     for row in reader:
    #     #         youtube_link = row[2].strip()
    #     #         print(".............",youtube_link)

    #     #     # Check if the link has already been processed
    #     #         if youtube_link in processed_links:
    #     #             print(f"Skipping already processed link: {youtube_link}")
    #     #             continue

    #     #         try:
    #     #             print(f"Processing link: {youtube_link}")
    #     #             youtube_video = YouTube(youtube_link)
    #     #             highest_resolution = youtube_helper.get_highest_resolution(youtube_video)

    #     #             if highest_resolution is not None:
    #     #                 row.append(highest_resolution)
    #     #                 csv_file.seek(0)
    #     #                 writer.writerows(reader)
    #     #                 csv_file.truncate()
    #     #                 processed_links.add(youtube_link)
    #     #                 print(f"Highest resolution for {youtube_link}: {highest_resolution}")
    #     #             else:
    #     #                 print(f"No valid video streams found for {youtube_link}")

    #     #         except Exception as e:
    #     #             print(f"Error processing {youtube_link}: {str(e)}")
