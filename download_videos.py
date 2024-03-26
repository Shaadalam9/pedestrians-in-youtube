import os
from pytube import YouTube
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
            youtube_object = YouTube(youtube_url)
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
            selected_stream.download(output_path, filename=f"{vid}.mp4") 
            logger.info("Started download of video {} entitled \"{}\" in resolution {}.", vid, youtube_object.title,
                        resolution)
    except Exception as e:
        logger.error("Error occurred {}.", e)
        return None


if __name__ == "__main__":
    df_mapping = pd.read_csv("mapping.csv")

    new_directory = common.get_configs('source_videos')
    if not os.path.exists(new_directory):
        # If it doesn't exist, create it
        os.makedirs(new_directory)
        logger.debug("Directory '{}' created successfully.".format(new_directory))
    else:
        logger.debug("Directory '{}' already exists.".format(new_directory))

    for index, row in df_mapping.iterrows():
        video_link = row['videos']
        my_city = row['city']
        my_condition = row['time_of_day']
        download_video_with_resolution(video_link, output_path=new_directory)
