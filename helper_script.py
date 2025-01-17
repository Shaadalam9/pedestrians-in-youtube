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

logger = CustomLogger(__name__)  # use custom logger

mapping = pd.read_csv(common.get_configs("mapping"))
confidence = common.get_configs("confidence")
render = common.get_configs("render")
line_thickness = common.get_configs("line_thickness")
show_labels = common.get_configs("show_labels")
show_conf = common.get_configs("show_conf")
display_frame_tracking = common.get_configs("display_frame_tracking")
output_path = common.get_configs("output_path")
save_annoted_img = common.get_configs("save_annoted_img")
delete_labels = common.get_configs("delete_labels")
delete_frames = common.get_configs("delete_frames")
display_frame_tracking = common.get_configs("display_frame_tracking")


class youtube_helper:

    def __init__(self, video_title=None):
        self.model = common.get_configs("model")
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

    def download_video_with_resolution(self, video_id, resolutions=["720p", "480p", "144p"], output_path="."):
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

            video_file_path = os.path.join(output_path, f"{video_id}.mp4")
            logger.info("Youtube video download in progress...")
            # Comment the below line to automatically download with video in "video" folder
            selected_stream.download(output_path, filename=f"{video_id}.mp4")

            logger.info(f"Download of '{youtube_object.title}' in {resolution} completed successfully.")
            self.video_title = youtube_object.title

            # Get the FPS of the video
            fps = self.get_video_fps(video_file_path)
            logger.info(f"The fps of '{youtube_object.title}' is '{fps}'")

            return video_file_path, video_id, resolution, fps

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

    def get_video_fps(self, video_file_path):
        try:
            # Open the video file using OpenCV
            video = cv2.VideoCapture(video_file_path)
            # Get FPS using OpenCV's `CAP_PROP_FPS` property
            fps = video.get(cv2.CAP_PROP_FPS)
            video.release()
            return round(fps, 0)
        except Exception as e:
            logger.error(f"Failed to retrieve FPS: {e}")
            return None

    @staticmethod
    def trim_video(input_path, output_path, start_time, end_time):
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)

        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        video_clip.close()

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

        # Append the DataFrame to the CSV file
        if not os.path.exists(output_csv):
            df.to_csv(output_csv, index=False, mode='w')  # If the CSV does not exist, create it
        else:
            df.to_csv(output_csv, index=False, mode='a', header=False)  # If it exists, append without header

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
            population_df.columns[0]: 'ISO_country',
            population_df.columns[2]: 'Year',
            population_df.columns[3]: 'Population'
        })

        # Divide population by 1000
        population_df['Population'] = population_df['Population'] / 1000

        return population_df

    @staticmethod
    def update_population_in_csv(data):

        # Ensure the required columns exist in the CSV
        if "ISO_country" not in data.columns:
            raise KeyError("The CSV file does not have a 'ISO_country' column.")

        if "population_country" not in data.columns:
            data["population_country"] = None  # Initialize the column if it doesn't exist

        # Get the latest population data
        latest_population = youtube_helper.get_latest_population()

        # Create a dictionary for quick lookup
        population_dict = dict(zip(latest_population['ISO_country'], latest_population['Population']))

        # Update the population_country column
        for index, row in data.iterrows():
            ISO_country = row["ISO_country"]
            population = population_dict.get(ISO_country, None)
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
                        max_fps_format = max(formats_720p, key=lambda fmt: fmt['fps'])
                        return max_fps_format['fps']

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
            if isinstance(existing_fps_list, str) and existing_fps_list.strip():
                try:
                    existing_fps = eval(existing_fps_list)  # Convert string to list
                except Exception as e:
                    logger.error(f"Error parsing existing_fps_list: {e}")
                    existing_fps = [None] * len(video_ids)
            else:
                existing_fps = [None] * len(video_ids)

            fps_values = []
            for i, video_id in enumerate(video_ids):
                video_id = video_id.strip()
                # Skip processing if the FPS value already exists
                if i < len(existing_fps) and existing_fps[i] is not None:
                    fps_values.append(existing_fps[i])
                    logger.info(f"Skipping video {video_id} as FPS is already available.")
                else:
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
            pd.DataFrame: DataFrame with ISO_country and the latest Traffic Mortality Rate.
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
            if 'ISO_country' not in df.columns or 'traffic_mortality' not in df.columns:
                logger.error("The required columns 'ISO_country' and 'traffic_mortality' are missing from the file.")
                return

            # Get the latest traffic mortality data
            traffic_df = youtube_helper.get_latest_traffic_mortality()

            # Merge the traffic mortality data with the existing DataFrame
            updated_df = pd.merge(df, traffic_df, on='ISO_country', how='left', suffixes=('', '_new'))

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
    def get_latest_geni_values():
        """
        Fetch the latest GINI index data from the World Bank.

        Returns:
            pd.DataFrame: DataFrame with ISO_country and the latest GINI index values.
        """
        # World Bank indicator for GINI index
        indicator = 'SI.POV.GINI'  # GINI index

        # Fetch the most recent data (mrv=1 for the most recent value)
        geni_data = wb.get_series(indicator, id_or_value='id', mrv=1)

        # Convert the data to a DataFrame
        geni_df = geni_data.reset_index()

        # Rename columns appropriately
        geni_df = geni_df.rename(columns={
            geni_df.columns[0]: 'ISO_country',
            geni_df.columns[2]: 'Year',
            geni_df.columns[3]: 'geni'
        })

        # Keep only the latest value for each country
        geni_df = geni_df.sort_values(by=['ISO_country', 'Year'],
                                      ascending=[True, False]).drop_duplicates(subset=['ISO_country'])

        return geni_df[['ISO_country', 'geni']]

    @staticmethod
    def fill_gini_data(df):
        """
        Fill the GINI index column in a CSV file using World Bank data.

        Args:
            file_path (str): Path to the input CSV file.
        """
        try:
            # Ensure the required column exists
            if 'geni' not in df.columns:
                logger.error("The required columns 'ISO_country' and 'geni' are missing from the file.")
                return

            # Get the latest GINI index data
            geni_df = youtube_helper.get_latest_geni_values()

            # Merge the GINI index data with the existing DataFrame
            updated_df = pd.merge(df, geni_df, on='ISO_country', how='left', suffixes=('', '_new'))

            # Update the geni column with the new data
            updated_df['geni'] = updated_df['geni_new'].combine_first(updated_df['geni'])

            # Drop the temporary column
            updated_df = updated_df.drop(columns=['geni_new'])

            # Save the updated DataFrame back to the same CSV file
            updated_df.to_csv(common.get_configs("mapping"), index=False)
            logger.info("Mapping file updated successfully with GINI value.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def prediction_mode(self):
        model = YOLO(self.model)
        model.predict(source=os.path.join(output_path, f"{self.video_title}.mp4"),
                      save=True, conf=confidence, save_txt=True,
                      show=render, line_width=line_thickness,
                      show_labels=show_labels, show_conf=show_conf)

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

        # Loop through the video frames
        frame_count = 0  # Variable to track the frame number
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:

                frame_count += 1  # Increment frame count
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, tracker='bytetrack.yaml',
                                      persist=True, conf=confidence,
                                      save=True, save_txt=True,
                                      line_width=line_thickness,
                                      show_labels=show_labels,
                                      show_conf=show_labels, show=render)

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

                youtube_helper.merge_txt_to_csv_dynamically(labels_path, output_csv_path, frame_count)

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
                                      thickness=line_thickness*5)

                except Exception:
                    pass

                # Display the annotated frame
                if display_frame_tracking:

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
