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
            logger.info(vid, start_time, end_time, time_of_day_value)

            # Attempt to download the video
            result = helper.download_video_with_resolution(video_id=vid, output_path=params.output_path)
            # result = None
            if result:
                video_file_path, video_title, resolution = result
                logger.info(f"Downloaded video: {video_file_path}")
                helper.set_video_title(video_title)
            else:
                # If download fails, check if the video already exists in the folder
                video_file_path = f"{params.output_path}/{vid}.mp4"
                if os.path.exists(video_file_path):
                    video_title = vid  # Assuming the video ID is the title for simplicity
                    logger.info(f"Video found: {video_file_path}")
                    helper.set_video_title(video_title)
                else:
                    logger.error(f"Video {vid} not found and download failed. Skipping this video.")
                    continue

            input_video_path = video_file_path
            output_video_path = f"{params.output_path}/{video_title}_mod.mp4"

            if start_time is None and end_time is None:
                logger.info("No trimming required")
            else:
                logger.info("Trimming in progress.......")
                # Some frames are missing in the last seconds
                end_time = end_time - 1
                helper.trim_video(input_video_path, output_video_path, start_time, end_time)
                os.remove(input_video_path)
                logger.info("Deleted the untrimmed video")
                os.rename(output_video_path, input_video_path)

            if params.prediction_mode:
                helper.prediction_mode()

            if params.tracking_mode:
                helper.tracking_mode(input_video_path, output_video_path)
                if params.need_annotated_video:
                    helper.create_video_from_images(params.frames_output_path, params.final_video_output_path, 30)

                data_folder = "data"
                os.makedirs(data_folder, exist_ok=True)
                os.rename(f"runs/detect/{vid}.csv", f"runs/detect/{vid}_{start_time}.csv")
                shutil.move(f"runs/detect/{vid}_{start_time}.csv", data_folder)

                shutil.rmtree("runs/detect/predict")

                if params.delete_runs_files:
                    shutil.rmtree("runs/detect")
                else:
                    helper.rename_folder(
                        "runs/detect", f"runs/{video_title}_{resolution}_{datetime.now()}"
                    )
                counter += 1

            if params.delete_youtube_video:
                os.remove(input_video_path)
