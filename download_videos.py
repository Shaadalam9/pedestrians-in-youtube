# by Shadab Alam
import os
from pytubefix import YouTube
from pytubefix.cli import on_progress
import pandas as pd
import common
from custom_logger import CustomLogger
from logmod import logs


logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def download_video_with_resolution(video_ids, resolutions=["2160p", "1440p", "1080p", "720p", "480p", "360p"],
                                   output_path="."):
    try:
        video_id = [id.strip() for id in video_ids.strip("[]").split(',')]
        for vid in video_id:
            # Check if the MP4 file already exists in the output folder
            mp4_file_path = os.path.join(output_path, f"{vid}.mp4")
            if os.path.exists(mp4_file_path):
                logger.debug(f"MP4 file for video with {vid} already exists in the folder.")
                continue
            youtube_url = f'https://www.youtube.com/watch?v={vid}'
            youtube_object = YouTube(youtube_url, on_progress_callback=on_progress)
            for resolution in resolutions:
                video_streams = youtube_object.streams.filter(res=f"{resolution}")
                if video_streams:
                    logger.debug(f"Found video {vid} in {resolution}.")
                    break

            if not video_streams:
                logger.debug(f"No {resolution} resolution available for {vid}.")
                return None

            selected_stream = video_streams[0]
            # Comment the below line to automatically download with video in "video" folder
            logger.info("Started download of video {} entitled \"{}\" in resolution {}.", vid, youtube_object.title,
                        resolution)
            selected_stream.download(output_path, filename=f"{vid}.mp4")
    except Exception as e:
        logger.error("Error occurred {}.", e)
        return None


if __name__ == "__main__":
    logger.info("Download of videos started.")
    df_mapping = pd.read_csv("mapping.csv")
    source_videos = common.get_configs('save_download_videos')  # folder for output of videos
    # Create folder for output
    if not os.path.exists(source_videos):
        os.makedirs(source_videos)
        logger.debug("Directory '{}' created successfully.".format(source_videos))
    else:
        logger.debug("Directory '{}' already exists.".format(source_videos))
    num_videos = 0  # counter of videos
    # Go over videos
    for index, row in df_mapping.iterrows():
        logger.info("Analysing videos for town={}, time_of_day={}.", row['city'], row['time_of_day'])
        download_video_with_resolution(row['videos'], output_path=source_videos)
    logger.info("Processed {} videos.", num_videos)
