# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from ultralytics import YOLO
from collections import defaultdict
import shutil
import numpy as np
import pandas as pd
import world_bank_data as wb
import yt_dlp
import pycountry
from pycountry_convert import country_name_to_country_alpha2, country_alpha2_to_continent_code
from custom_logger import CustomLogger
import common
import ast
import subprocess
import sys
import logging
from tqdm import tqdm
import datetime
import json


logger = CustomLogger(__name__)  # use custom logger
logging.getLogger("ultralytics").setLevel(logging.ERROR)  # Show only errors

mapping = pd.read_csv(common.get_configs("mapping"))
confidence = common.get_configs("confidence")

display_frame_tracking = common.get_configs("display_frame_tracking")
output_path = common.get_configs("videos")
save_annoted_img = common.get_configs("save_annoted_img")
delete_labels = common.get_configs("delete_labels")
delete_frames = common.get_configs("delete_frames")

# Consts
LINE_TICKNESS = 1
RENDER = False
SHOW_LABELS = False
SHOW_CONF = False

# logging of attempts to upgrade packages
UPGRADE_LOG_FILE = "upgrade_log.json"


class Youtube_Helper:

    def __init__(self, video_title=None):
        """
        Initialises a new instance of the class.

        Parameters:
            video_title (str, optional): The title of the video. Defaults to None.

        Instance Variables:
            self.model (str): The model configuration loaded from common.get_configs("model").
            self.resolution (str): The video resolution. Initialized as None and set later when needed.
            self.video_title (str): The title of the video.
        """
        self.model = common.get_configs("model")
        self.resolution = None
        self.video_title = video_title

    def set_video_title(self, title):
        """
        Sets the video title for the instance.

        Parameters:
            title (str): The new title for the video.
        """
        self.video_title = title

    @staticmethod
    def rename_folder(old_name, new_name):
        """
        Renames a folder from old_name to new_name.

        Parameters:
            old_name (str): The current name (or path) of the folder.
            new_name (str): The new name (or path) to assign to the folder.

        Error Handling:
            - Logs an error if the folder with old_name is not found.
            - Logs an error if a folder with new_name already exists.
        """
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            logger.error(f"Error: Folder '{old_name}' not found.")
        except FileExistsError:
            logger.error(f"Error: Folder '{new_name}' already exists.")

    @staticmethod
    def load_upgrade_log():
        """Loads the upgrade log from a file."""
        if not os.path.exists(UPGRADE_LOG_FILE):
            return {}
        try:
            with open(UPGRADE_LOG_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def save_upgrade_log(log_data):
        """Saves the upgrade log to a file."""
        with open(UPGRADE_LOG_FILE, "w") as file:
            json.dump(log_data, file)

    @staticmethod
    def was_upgraded_today(package_name):
        """Checks if the package was attempted to be upgraded today."""
        log_data = Youtube_Helper.load_upgrade_log()
        today = datetime.date.today().isoformat()
        return log_data.get(package_name) == today

    @staticmethod
    def mark_as_upgraded(package_name):
        """Logs that a package upgrade was attempted today."""
        log_data = Youtube_Helper.load_upgrade_log()
        log_data[package_name] = datetime.date.today().isoformat()
        Youtube_Helper.save_upgrade_log(log_data)

    @staticmethod
    def upgrade_package_if_needed(package_name):
        """
        Upgrades a given Python package using pip if it hasn't been attempted today.

        Parameters:
            package_name (str): The name of the package to upgrade.
        """
        if Youtube_Helper.was_upgraded_today(package_name):
            logging.debug(f"{package_name} upgrade already attempted today. Skipping.")
            return

        try:
            logging.info(f"Upgrading {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            logging.info(f"{package_name} upgraded successfully.")
            Youtube_Helper.mark_as_upgraded(package_name)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to upgrade {package_name}: {e}")
            Youtube_Helper.mark_as_upgraded(package_name)  # still log it to avoid retrying

    def download_video_with_resolution(self, vid, resolutions=["720p", "480p", "360p", "144p"], output_path="."):
        """
        Downloads a YouTube video in one of the specified resolutions and returns video details.

        This function attempts to download the video using the pytubefix/YouTube method.

        Parameters:
            vid (str): The YouTube video ID.
            resolutions (list of str, optional): A list of preferred video resolutions.
            output_path (str, optional): The directory where the video will be downloaded.

        Returns:
            tuple or None: A tuple (video_file_path, vid, resolution, fps) if successful,
                            or None if methods fail.
        """
        try:
            # Optionally upgrade pytubefix (if configured and it is Monday)
            if common.get_configs("update_package") and datetime.datetime.today().weekday() == 0:
                Youtube_Helper.upgrade_package_if_needed("pytube")
                Youtube_Helper.upgrade_package_if_needed("pytubefix")

            # Construct the YouTube URL.
            youtube_url = f"https://www.youtube.com/watch?v={vid}"

            # Create a YouTube object using the provided client configuration.
            if common.get_configs("need_authentication"):
                youtube_object = YouTube(youtube_url,
                                         common.get_configs('client'),
                                         use_oauth=True,
                                         allow_oauth_cache=True,
                                         on_progress_callback=on_progress)
            else:
                youtube_object = YouTube(youtube_url,
                                         common.get_configs('client'),
                                         on_progress_callback=on_progress)

            selected_stream = None
            selected_resolution = None

            # Iterate over the preferred resolutions to find a matching stream.
            for resolution in resolutions:
                # Filter the streams for the current resolution.
                video_streams = youtube_object.streams.filter(res=resolution)
                if video_streams:
                    selected_resolution = resolution
                    logger.debug(f"Found video {vid} in {resolution}.")
                    # Use the first available stream.
                    if hasattr(video_streams, 'first'):
                        selected_stream = video_streams.first()
                    else:
                        selected_stream = video_streams[0]
                    break

            if not selected_stream:
                logger.error(f"{vid}: no stream available for video in the specified resolutions.")
                return None

            # Construct the file path for the downloaded video.
            video_file_path = os.path.join(output_path, f"{vid}.mp4")
            logger.info(f"{vid}: download in {selected_resolution} started with pytube.")

            # Download the video.
            selected_stream.download(output_path, filename=f"{vid}.mp4")
            self.video_title = youtube_object.title
            fps = self.get_video_fps(video_file_path)
            logger.info(f"{vid}: FPS={fps}.")

            return video_file_path, vid, selected_resolution, fps

        except Exception as e:
            logger.error(f"{vid}: pytubefix download method failed: {e}")
            logger.info(f"{vid}: falling back to yt_dlp method.")

        # ----- Fallback method: using yt_dlp -----
        try:
            # Optionally upgrade yt_dlp (if the configuration requires it and it is Monday)
            if common.get_configs("update_package") and datetime.datetime.today().weekday() == 0:
                Youtube_Helper.upgrade_package_if_needed("yt_dlp")

            # Construct the YouTube URL.
            youtube_url = f"https://www.youtube.com/watch?v={vid}"

            # Extract video information (including available formats) without downloading.
            extract_opts = {
                'skip_download': True,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(extract_opts) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=False)

            available_formats = info_dict.get("formats", [])  # type: ignore
            selected_format_str = None
            selected_resolution = None

            # Iterate over the preferred resolutions.
            for res in resolutions:
                try:
                    res_height = int(res.rstrip("p"))
                except ValueError:
                    continue

                # Check for a video-only stream (no audio).
                video_only_found = any(
                    fmt for fmt in available_formats
                    if fmt.get("height") == res_height and fmt.get("acodec") == "none"
                )
                if video_only_found:
                    selected_format_str = f"bestvideo[height={res_height}]"
                    selected_resolution = res
                    logger.info(f"{vid}: found video-only format in {res}.")
                    break

                # Otherwise, check for any stream at that resolution.
                progressive_found = any(
                    fmt for fmt in available_formats
                    if fmt.get("height") == res_height
                )
                if progressive_found:
                    selected_format_str = f"best[height={res_height}]"
                    selected_resolution = res
                    logger.info(f"{vid}: found progressive format in {res}. Audio will be removed.")
                    break

            if not selected_format_str:
                logger.error(f"{vid}: no stream available in the specified resolutions.")
                # Raise an exception to trigger the fallback method.
                raise Exception(f"{vid}: no stream available via yt_dlp")
            po_token = common.get_secrets("po_token")
            # Set download options.
            download_opts = {
                'format': selected_format_str,
                'outtmpl': os.path.join(output_path, f"{vid}.%(ext)s"),
                'quiet': True,
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4'
                }],
                'postprocessor_args': ['-an'],
                'http_headers': {
                    'Cookie': f'pot={po_token}'
                },
                'extractor_args': {
                    'youtube': {
                        'po_token': f'web.gvs+{po_token}'
                    }
                },
                'cookiesfrombrowser': ('chrome',)
            }

            logger.info(f"{vid}: download in {selected_resolution} started with yt_dlp.")
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                ydl.download([youtube_url])

            # Final output file path (assuming the postprocessor outputs an MP4 file).
            video_file_path = os.path.join(output_path, f"{vid}.mp4")
            self.video_title = info_dict.get("title")  # type: ignore
            fps = self.get_video_fps(video_file_path)
            logger.info(f"FPS of {vid}: {fps}.")

            return video_file_path, vid, selected_resolution, fps

        except Exception as e:
            logger.error(f"{vid}: yt_dlp download method failed: {e}.")
            return None

    def get_video_fps(self, video_file_path):
        """
        Retrieves the frames per second (FPS) of a video file using OpenCV.

        Parameters:
            video_file_path (str): The file path to the video whose FPS is to be determined.

        Returns:
            int or None: The rounded FPS value of the video if successful; otherwise, returns None if an error occurs.

        The function performs the following steps:
            1. Opens the video file using OpenCV's VideoCapture.
            2. Retrieves the FPS using the CAP_PROP_FPS property.
            3. Rounds the FPS value to the nearest integer.
            4. Releases the video resource.
            5. Returns the rounded FPS value, or None if an exception is encountered.
        """
        try:
            # Open the video file using OpenCV
            video = cv2.VideoCapture(video_file_path)

            # Retrieve the FPS using OpenCV's CAP_PROP_FPS property
            fps = video.get(cv2.CAP_PROP_FPS)

            # Release the video resource
            video.release()

            # Return the FPS rounded to the nearest integer
            return round(fps, 0)
        except Exception as e:
            # Log an error message if FPS retrieval fails
            logger.error(f"Failed to retrieve FPS: {e}")
            return None

    @staticmethod
    def trim_video(input_path, output_path, start_time, end_time):
        """
        Trims a segment from a video and saves the result to a specified file.

        Parameters:
            input_path (str): The file path to the original video.
            output_path (str): The destination file path where the trimmed video will be saved.
            start_time (float or str): The start time for the trimmed segment. This can be specified in seconds
                                       or in a time format recognized by MoviePy.
            end_time (float or str): The end time for the trimmed segment. Similar to start_time, it can be in seconds
                                     or another supported time format.

        Returns:
            None

        The function performs the following steps:
            1. Loads the original video using MoviePy's VideoFileClip.
            2. Creates a subclip from the original video based on the provided start_time and end_time.
            3. Writes the subclip to the output_path using the H.264 video codec and AAC audio codec.
            4. Closes the video file to free up resources.
        """
        # Load the video and create a subclip using the provided start and end times.
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)

        # Write the subclip to the specified output file using the 'libx264' codec for video and 'aac' for audio.
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close the video clip to release any resources used.
        video_clip.close()

    @staticmethod
    def detect_gpu():
        """
        Detects whether an NVIDIA or Intel GPU is available and returns the appropriate FFmpeg encoder.

        Returns:
            str: 'hevc_nvenc' for NVIDIA, 'hevc_qsv' for Intel, or None if no compatible GPU is found.
        """
        try:
            # Check for NVIDIA GPU
            nvidia_check = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if nvidia_check.returncode == 0:
                return "hevc_nvenc"

            # Check for Intel QuickSync GPU
            intel_check = subprocess.run(["vainfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "Intel" in intel_check.stdout:
                return "hevc_qsv"

        except FileNotFoundError:
            pass  # Command not found, meaning the hardware isn't available or the driver isn't installed

        return None  # No compatible GPU found

    @staticmethod
    def compress_video(input_path, codec="libx265", preset="slow", crf=17):
        """
        Compresses a video using codec=codec.

        Args:
            input_path (str): Path to the input video file.
            codec (str, optional): Codec to use. Use H.265 by default.
            preset (str, optional): Value for preset.
            crf (int, optional): Value for crf. 17 is supposed to keep good quality with high level of compression.

        Returns:
            str: Path to the compressed video.

        Raises:
            e: error.
            FileNotFoundError: If the input video file does not exist.
            RuntimeError: If the compression process fails.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Extract filename and create output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(common.root_dir, filename)

        # Detect available GPU and set appropriate encoder
        codec_hw = Youtube_Helper.detect_gpu()
        if codec_hw:
            codec = codec_hw  # Use detected hardware

        # Construct ffmpeg command
        command = [
            "ffmpeg",
            "-i", input_path,  # Input file
            "-c:v", codec,  # Use appropriate codec
            "-preset", preset,  # Compression speed/efficiency tradeoff
            "-crf", str(crf),  # Constant Rate Factor (lower = better quality, larger file)
            output_path,  # Temporary output file
            "-progress", "pipe:1",  # Enables real-time progress output
            "-nostats"  # Suppresses extra logs
        ]

        try:
            # Run ffmpeg command
            video_id = Youtube_Helper.extract_youtube_id(input_path)
            logger.info(f"Started compression of {video_id} with {codec} codec. Current size={os.path.getsize(input_path)}.")  # noqa: E501
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            logger.info(f"Finished compression of {video_id} with {codec} codec. New size={os.path.getsize(output_path)}.")  # noqa: E501
            # Replace the original file with the compressed file
            shutil.move(output_path, input_path)
        except Exception as e:
            # Clean up temporary file in case of unexpected errors
            if os.path.exists(output_path):
                os.remove(output_path)
            logger.error(f"Video compression failed: {e.stderr.decode()}. Using uncompressed file.")  # type: ignore

    @staticmethod
    def extract_youtube_id(file_path):
        """
        Extracts the YouTube video ID from a given file path.

        Args:
            file_path (str): The full path of the video file.

        Returns:
            str: The extracted YouTube ID.

        Raises:
            ValueError: If no valid YouTube ID is found.
        """
        filename = os.path.basename(file_path)  # Get only the filename
        youtube_id, ext = os.path.splitext(filename)  # Remove the file extension

        if not youtube_id or len(youtube_id) < 5:  # Basic validation
            raise ValueError("Invalid YouTube ID extracted.")

        return youtube_id

    @staticmethod
    def create_video_from_images(image_folder, output_video_path, frame_rate):
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
        df['Frame Count'] = frame_count

        # Append the DataFrame to the CSV file
        if not os.path.exists(output_csv):
            df.to_csv(output_csv, index=False, mode='w')  # If the CSV does not exist, create it
        else:
            df.to_csv(output_csv, index=False, mode='a', header=False)  # If it exists, append without header

    @staticmethod
    def delete_folder(folder_path):
        """
        Deletes the folder and all its contents recursively.

        Parameters:
            folder_path (str): The path of the folder to delete.

        Returns:
            bool: True if the folder was successfully deleted, False otherwise.
        """
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                logger.info(f"Folder '{folder_path}' deleted successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to delete folder '{folder_path}': {e}")
                return False
        else:
            logger.info(f"Folder '{folder_path}' does not exist.")
            return False

    @staticmethod
    def check_missing_mapping(mapping):
        for index, row in mapping.iterrows():
            video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
            start_times = ast.literal_eval(row["start_time"])
            for vid_index, (vid, start_times_list) in enumerate(zip(video_ids, start_times)):
                for start_time in start_times_list:
                    file_name = f'{vid}_{start_time}.csv'
                    file_path = os.path.join(common.get_configs("data"), file_name)
                    # Check if the file exists
                    if os.path.isfile(file_path):
                        pass
                    else:
                        logger.info(f"The file '{file_name}' does not exist.")

    @staticmethod
    def get_iso_alpha_3(country_name, existing_iso):
        try:
            return pycountry.countries.lookup(country_name).alpha_3
        except LookupError:
            if country_name.strip().upper() == "KOSOVO":
                return "XKX"  # User-assigned code for Kosovo
            return existing_iso if existing_iso else None

    @staticmethod
    def get_latest_population():
        # Search for the population indicator
        indicator = 'SP.POP.TOTL'  # Total Population (World Bank indicator code)

        # Fetch the latest population data
        population_data = wb.get_series(indicator, id_or_value='id', mrv=1)

        # Convert the data to a DataFrame
        population_df = population_data.reset_index()

        # Rename columns appropriately
        population_df = population_df.rename(columns={
            population_df.columns[0]: 'iso3',
            population_df.columns[2]: 'Year',
            population_df.columns[3]: 'Population'
        })

        # Divide population by 1000
        population_df['Population'] = population_df['Population'] / 1000

        return population_df

    @staticmethod
    def update_population_in_csv(data):

        # Ensure the required columns exist in the CSV
        if "iso3" not in data.columns:
            raise KeyError("The CSV file does not have a 'iso3' column.")

        if "population_country" not in data.columns:
            data["population_country"] = None  # Initialize the column if it doesn't exist

        # Get the latest population data
        latest_population = Youtube_Helper.get_latest_population()

        # Create a dictionary for quick lookup
        population_dict = dict(zip(latest_population['iso3'], latest_population['Population']))

        # Update the population_country column
        for index, row in data.iterrows():
            iso3 = row["iso3"]
            population = population_dict.get(iso3, None)
            data.at[index, "population_country"] = population  # Always update with the latest population data

        # Save the updated DataFrame back to the same CSV
        data.to_csv(common.get_configs("mapping"), index=False)
        logger.info("Mapping file updated sucessfully with country population.")

    @staticmethod
    def get_continent_from_country(country):
        """
        Returns the continent based on the country name using pycountry_convert.
        """
        try:
            # Convert country name to ISO Alpha-2 code
            alpha2_code = country_name_to_country_alpha2(country)
            # Convert ISO Alpha-2 code to continent code
            continent_code = country_alpha2_to_continent_code(alpha2_code)
            # Map continent codes to continent names
            continent_map = {
                "AF": "Africa",
                "AS": "Asia",
                "EU": "Europe",
                "NA": "North America",
                "SA": "South America",
                "OC": "Oceania",
                "AN": "Antarctica"
            }
            return continent_map.get(continent_code, "Unknown")
        except KeyError:
            return "Unknown"

    @staticmethod
    def update_csv_with_fps(df):
        """
        Updates the existing CSV file by adding a new column 'fps_list' with the FPS values for each video's IDs.

        Args:
            file_path (str): The path to the CSV file to be updated.
        """
        def get_fps(video_id):
            try:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                ydl_opts = {'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    if not info:
                        return None

                    # Extract formats with 720p resolution and FPS information
                    formats_720p = [
                        fmt
                        for fmt in info.get('formats', [])
                        if fmt.get('height') == 720 and 'fps' in fmt
                    ]

                    if formats_720p:
                        # Select the format with the highest FPS
                        first_fps_format = formats_720p[0]
                        return first_fps_format['fps']

                    # Fallback: No 720p formats with FPS information
                    logger.warning(f"No valid 720p formats with FPS found for video {video_id}.")
                    return None
            except yt_dlp.utils.DownloadError:
                logger.error(f"Video {video_id} not found or unavailable.")
                return None
            except Exception as e:
                logger.error(f"Error fetching FPS for video {video_id}: {e}")
                return None

        def process_videos(video_ids, existing_fps_list):
            video_ids = video_ids.strip("[]").split(",")

            # Ensure existing_fps_list is a valid string or initialize as an empty list
            # if isinstance(existing_fps_list, str) and existing_fps_list.strip():
            #     try:
            #         existing_fps = eval(existing_fps_list)  # Convert string to list
            #     except Exception as e:
            #         logger.error(f"Error parsing existing_fps_list: {e}")
            #         existing_fps = [None] * len(video_ids)
            # else:
            #     existing_fps = [None] * len(video_ids)

            fps_values = []
            for i, video_id in enumerate(video_ids):
                video_id = video_id.strip()
                # Skip processing if the FPS value already exists
                # if i < len(existing_fps) and existing_fps[i] is not None:
                #     fps_values.append(existing_fps[i])
                #     logger.info(f"Skipping video {video_id} as FPS is already available.")
                # else:
                fps = get_fps(video_id)
                fps_values.append(fps)
            return str(fps_values)
        # Process the 'videos' column and add/update the 'fps_list' column
        if 'fps_list' not in df.columns:
            df['fps_list'] = None

        for index, row in df.iterrows():
            existing_fps_list = row['fps_list']
            df.at[index, 'fps_list'] = process_videos(row['videos'], existing_fps_list)

        # Save the updated DataFrame back to the same file
        df.to_csv(common.get_configs("mapping"), index=False)
        logger.info("Mapping file updated successfully with FPS data.")

    @staticmethod
    def get_upload_date(video_id):
        try:
            # Construct YouTube URL from video ID
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            # Create YouTube object
            yt = YouTube(video_url)

            # Fetch upload date
            upload_date = yt.publish_date

            if upload_date:
                # Format the date as ddmmyyyy
                return upload_date.strftime('%d%m%Y')
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def get_latest_traffic_mortality():
        """
        Fetch the latest traffic mortality data from the World Bank.

        Returns:
            pd.DataFrame: DataFrame with iso3 and the latest Traffic Mortality Rate.
        """
        # World Bank indicator for traffic mortality rate
        indicator = 'SH.STA.TRAF.P5'  # Road traffic deaths per 100,000 people

        # Fetch the most recent data (mrv=1 for the most recent value)
        traffic_mortality_data = wb.get_series(indicator, id_or_value='id', mrv=1)

        # Convert the data to a DataFrame
        traffic_df = traffic_mortality_data.reset_index()

        # Rename columns appropriately
        traffic_df = traffic_df.rename(columns={
            traffic_df.columns[0]: 'ISO_country',
            traffic_df.columns[2]: 'Year',
            traffic_df.columns[3]: 'traffic_mortality'
        })

        # Keep only the latest value for each country
        traffic_df = traffic_df.sort_values(by=['ISO_country', 'Year'],
                                            ascending=[True, False]).drop_duplicates(subset=['ISO_country'])

        # Add default value for XKX
        traffic_df.loc[traffic_df['ISO_country'] == 'XKX', 'traffic_mortality'] = 7.4

        return traffic_df[['ISO_country', 'traffic_mortality']]

    @staticmethod
    def fill_traffic_mortality(df):
        """
        Fill the traffic mortality rate column in a CSV file using World Bank data.

        Args:
            file_path (str): Path to the input CSV file.
        """
        try:
            # Ensure the required columns exist
            if 'iso3' not in df.columns or 'traffic_mortality' not in df.columns:
                logger.error("The required columns 'iso3' and 'traffic_mortality' are missing from the file.")
                return

            # Get the latest traffic mortality data
            traffic_df = Youtube_Helper.get_latest_traffic_mortality()

            # Merge the traffic mortality data with the existing DataFrame
            updated_df = pd.merge(df, traffic_df, on='iso3', how='left', suffixes=('', '_new'))

            # Update the traffic_mortality column with the new data
            updated_df['traffic_mortality'] = updated_df['traffic_mortality_new'].combine_first(
                updated_df['traffic_mortality'])

            # Drop the temporary column
            updated_df = updated_df.drop(columns=['traffic_mortality_new'])

            # Save the updated DataFrame back to the same CSV file
            updated_df.to_csv(common.get_configs("mapping"), index=False)
            logger.info("Mapping file updated successfully with traffic mortality rate.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

    @staticmethod
    def get_latest_gini_values():
        """
        Fetch the latest GINI index data from the World Bank.

        Returns:
            pd.DataFrame: DataFrame with iso3 and the latest GINI index values.
        """
        # World Bank indicator for GINI index
        indicator = 'SI.POV.GINI'  # GINI index

        # Fetch the most recent data (mrv=1 for the most recent value)
        geni_data = wb.get_series(indicator, id_or_value='id', mrv=1)

        # Convert the data to a DataFrame
        geni_df = geni_data.reset_index()

        # Rename columns appropriately
        geni_df = geni_df.rename(columns={
            geni_df.columns[0]: 'iso3',
            geni_df.columns[2]: 'Year',
            geni_df.columns[3]: 'gini'
        })

        # Keep only the latest value for each country
        geni_df = geni_df.sort_values(by=['iso3', 'Year'],
                                      ascending=[True, False]).drop_duplicates(subset=['iso3'])

        return geni_df[['iso3', 'gini']]

    @staticmethod
    def fill_gini_data(df):
        """
        Fill the GINI index column in a CSV file using World Bank data.

        Args:
            file_path (str): Path to the input CSV file.
        """
        try:
            # Ensure the required column exists
            if 'gini' not in df.columns:
                logger.error("The required columns 'iso3' and/or 'gini' are missing from the file.")
                return

            # Get the latest GINI index data
            geni_df = Youtube_Helper.get_latest_gini_values()

            # Merge the GINI index data with the existing DataFrame
            updated_df = pd.merge(df, geni_df, on='iso3', how='left', suffixes=('', '_new'))

            # Update the geni column with the new data
            updated_df['gini'] = updated_df['gini_new'].combine_first(updated_df['gini'])

            # Drop the temporary column
            updated_df = updated_df.drop(columns=['gini_new'])

            # Save the updated DataFrame back to the same CSV file
            updated_df.to_csv(common.get_configs("mapping"), index=False)
            logger.info("Mapping file updated successfully with GINI value.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def prediction_mode(self):
        model = YOLO(self.model)
        model.predict(source=os.path.join(output_path, f"{self.video_title}.mp4"),
                      save=True,
                      conf=confidence,
                      ave_txt=True,
                      show=RENDER,
                      line_width=1,
                      show_labels=SHOW_LABELS,
                      show_conf=SHOW_CONF)

    def tracking_mode(self, input_video_path, output_video_path, video_fps=25):
        model = YOLO(self.model)
        cap = cv2.VideoCapture(input_video_path)

        # Store the track history
        track_history = defaultdict(lambda: [])

        # Output paths for frames, txt files, and final video
        frames_output_path = os.path.join("runs", "detect", "frames")
        annotated_frame_output_path = os.path.join("runs", "detect", "annotated_frames")
        txt_output_path = os.path.join("runs", "detect", "labels")
        text_filename = os.path.join("runs", "detect", "track", "labels", "image0.txt")
        display_video_output_path = os.path.join("runs", "detect", "display_video.mp4")

        # Create directories if they don't exist
        os.makedirs(frames_output_path, exist_ok=True)
        os.makedirs(txt_output_path, exist_ok=True)
        os.makedirs(annotated_frame_output_path, exist_ok=True)

        # Initialize a VideoWriter for the final video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore

        if display_frame_tracking:
            display_video_writer = cv2.VideoWriter(display_video_output_path,
                                                   fourcc, video_fps, (int(cap.get(3)), int(cap.get(4))))

        # Open video and get total frames
        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print("Warning: Could not determine total frames. Progress bar may not work correctly.")
            total_frames = None  # Prevent tqdm from setting a fixed length

        # Setup progress bar
        progress_bar = tqdm(total=total_frames, unit="frames", dynamic_ncols=True)

        # Loop through the video frames
        frame_count = 0  # Variable to track the frame number
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:

                frame_count += 1  # Increment frame count
                # Run YOLO tracking on the frame, persisting tracks between frames
                results = model.track(frame,
                                      tracker='bytetrack.yaml',
                                      persist=True,
                                      conf=confidence,
                                      save=True,
                                      save_txt=True,
                                      line_width=LINE_TICKNESS,
                                      show_labels=SHOW_LABELS,
                                      show_conf=SHOW_CONF,
                                      show=RENDER,
                                      verbose=False)

                # Update progress bar
                progress_bar.update(1)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()  # type: ignore
                if boxes.size(0) == 0:
                    with open(text_filename, 'w') as file:   # noqa: F841
                        pass

                try:
                    track_ids = results[0].boxes.id.int().cpu().tolist()  # type: ignore

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()

                # Save annotated frame to file
                    if save_annoted_img:
                        frame_filename = os.path.join(annotated_frame_output_path, f"frame_{frame_count}.jpg")
                        cv2.imwrite(frame_filename, annotated_frame)

                except Exception:
                    pass

                # Save txt file with bounding box information
                with open(text_filename, 'r') as text_file:
                    data = text_file.read()
                new_txt_file_name = os.path.join("runs", "detect", "labels", f"label_{frame_count}.txt")
                with open(new_txt_file_name, 'w') as new_file:
                    new_file.write(data)

                labels_path = os.path.join("runs", "detect", "labels")
                output_csv_path = os.path.join("runs", "detect", f"{self.video_title}.csv")

                Youtube_Helper.merge_txt_to_csv_dynamically(labels_path, output_csv_path, frame_count)

                os.remove(text_filename)
                if delete_labels is True:
                    os.remove(os.path.join("runs", "detect", "labels", f"label_{frame_count}.txt"))

                # save the labelled image
                if delete_frames is False:
                    image_filename = os.path.join("runs", "detect", "track", "image0.jpg")
                    new_img_file_name = os.path.join("runs", "detect", "frames", f"frame_{frame_count}.jpg")
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
                                      thickness=LINE_TICKNESS*5)

                except Exception:
                    pass

                # Display the annotated frame
                if display_frame_tracking:

                    cv2.imshow("YOLOv11 Tracking", annotated_frame)
                    display_video_writer.write(annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
        progress_bar.close()
