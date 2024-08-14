import params as params
import shutil
import os
from datetime import datetime
from helper_script import youtube_helper
import pandas as pd
from custom_logger import CustomLogger
from logmod import logs
import ast

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger
helper = youtube_helper()
counter = 0

# Load the CSV file
df = pd.read_csv(params.input_csv_file)

for index, row in df.iterrows():
    video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
    start_times = ast.literal_eval(row["start_time"])
    end_times = ast.literal_eval(row["end_time"])
    time_of_day = ast.literal_eval(row["time_of_day"])

    for vid, start_times_list, end_times_list, time_of_day_list in zip(video_ids, start_times, end_times, time_of_day):
        for start_time, end_time, time_of_day_value in zip(start_times_list, end_times_list, time_of_day_list):
            print(vid, start_time, end_time, time_of_day_value)

            # Attempt to download the video
            result = helper.download_video_with_resolution(video_id=vid, output_path=params.output_path)
            if result:
                video_file_path, video_title, resolution = result
                print(f"Downloaded video: {video_file_path}")
            else:
                # If download fails, check if the video already exists in the folder
                video_file_path = f"{params.output_path}/{vid}.mp4"
                if os.path.exists(video_file_path):
                    video_title = vid  # Assuming the video ID is the title for simplicity
                    print(f"Video found: {video_file_path}")
                else:
                    logger.error(f"Video {vid} not found and download failed. Skipping this video.")
                    continue

            input_video_path = video_file_path
            output_video_path = f"{params.output_path}/{video_title}_mod.mp4"

            if start_time is None and end_time is None:
                print("No trimming required")
            else:
                print("Trimming in progress.......")
                # Some frames are missing in the last seconds
                end_time = end_time - 1
                helper.trim_video(input_video_path, output_video_path, start_time, end_time)
                os.remove(input_video_path)
                print("Deleted the untrimmed video")
                os.rename(output_video_path, input_video_path)

            print(f"{video_title}_{resolution}")

            if params.prediction_mode:
                helper.prediction_mode()

            if params.tracking_mode:
                helper.tracking_mode(input_video_path, output_video_path)
                if params.need_annotated_video:
                    helper.create_video_from_images(params.frames_output_path, params.final_video_output_path, 30)

                data_folder = "data"
                os.makedirs(data_folder, exist_ok=True)
                helper.merge_txt_files(params.txt_output_path, f"data/{video_title}_{start_time}.csv")

                shutil.rmtree("runs/detect/predict")
                if params.delete_frames:
                    shutil.rmtree("runs/detect/frames")

                helper.rename_folder(
                    "runs/detect", f"runs/{video_title}_{resolution}_{datetime.now()}"
                )
                counter += 1

            if params.delete_youtube_video:
                os.remove(input_video_path)
