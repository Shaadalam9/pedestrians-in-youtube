# compare one or multiple metrics across countries (cross-cultural effects)
#   1. presence of eye contact/hand gestures
#   2. amount of time before making the crossing decision
#   3. speed of crossing
#   4. hesitation (tbd)


import pandas as pd
import numpy as np
import params as params
from pytube import YouTube
from ultralytics import YOLO

def download_video_with_resolution(youtube_url, resolution="720p", output_path="."):
    try:
        youtube_object = YouTube(youtube_url)    # Create a YouTube object

        video_streams = youtube_object.streams.filter(res=f"{resolution}").all()


        if not video_streams:
            print(f"No {resolution} resolution available for '{youtube_object.title}'.")
            return None

        selected_stream = video_streams[0]

        video_file_path = f"{output_path}/{youtube_object.title}_{resolution}.mp4"
        selected_stream.download(output_path, filename=f"{youtube_object.title}_{resolution}.mp4")

        print(f"Download of '{youtube_object.title}' in {resolution} completed successfully.")
        
        return video_file_path, youtube_object.title
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


result = download_video_with_resolution(params.y_tube_link, resolution=params.resolution, output_path=params.output_path)

if result:
    video_file_path, video_title = result
    print(f"Video title: {video_title}")
    print(f"Video saved at: {video_file_path}")
else:
    print("Download failed.")



model = YOLO(params.model)
print(f"{video_title}_{params.resolution}")
model.predict(source = f"{video_title}_{params.resolution}.mp4", save= True, conf= params.confidence, save_txt=False, show=params.render)



