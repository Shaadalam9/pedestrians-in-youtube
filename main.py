import params as params
import shutil
import os
from datetime import datetime
from helper_script import youtube_helper
import pandas as pd

ffmpeg_path = "/Users/salam/anaconda3/envs/dashcam/bin/ffmpeg"  # Replace this with the actual path
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
helper = youtube_helper()
counter = 0

#Download the highest quality available videos quality value and update it in the csv file
df = pd.read_csv(params.input_csv_file)

youtube_links = df["Youtube link"]

for link in youtube_links:
    #Download the youtube video to the local system
    result = helper.download_video_with_resolution(youtube_url = link, output_path=params.output_path)
                                               
    if result:
        video_file_path, video_title, resolution = result
        print(f"Video title: {video_title}")
        print(f"Video saved at: {video_file_path}")
    else:
        print("Download failed.")


    input_video_path = f"{params.output_path}/{video_title}_{resolution}.mp4"
    output_video_path = f"{params.output_path}/{video_title}_{resolution}_mod.mp4"

    #Trimming of video (if required)
    start_time = params.trim_start
    end_time = params.trim_end  

    if start_time == None and end_time == None:
        print("No trimming required")
    else:
        print("Trimming in progress.......")
        helper.trim_video(input_video_path, output_video_path, start_time, end_time)

    os.remove(f"{params.output_path}/{video_title}_{resolution}.mp4")
    print("Deleted the untrimmed video")
    os.rename(output_video_path, input_video_path)

    print(f"{video_title}_{resolution}")

    if params.prediction_mode:
        helper.prediction_mode()

    if params.tracking_mode:
        helper.tracking_mode(input_video_path, output_video_path)
        helper.create_video_from_images(params.frames_output_path,params.final_video_output_path,30)

        helper.merge_txt_files(params.txt_output_path, params.output_merged_csv)

        shutil.rmtree("runs/detect/predict")
        helper.rename_folder("runs/detect", f"runs/{video_title}_{resolution}_{datetime.now()}")
        counter += 1

