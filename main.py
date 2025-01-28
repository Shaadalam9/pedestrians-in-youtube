import shutil
import os
from datetime import datetime
from helper_script import youtube_helper
import pandas as pd
from custom_logger import CustomLogger
from logmod import logs
import ast
import common

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger
helper = youtube_helper()
counter = 0

# Load the config file
mapping = pd.read_csv(common.get_configs("mapping"))
output_path = common.get_configs("output_path")
frames_output_path = common.get_configs("frames_output_path")
final_video_output_path = common.get_configs("final_video_output_path")
delete_runs_files = common.get_configs("delete_runs_files")
delete_youtube_video = common.get_configs("delete_youtube_video")
data_folder = common.get_configs("data")
countries_analyse = common.get_configs("countries_analyse")

if common.get_configs("check_for_download_csv_file"):
    helper.check_for_download_csv_file(mapping)

if common.get_configs("update_ISO_code"):
    # Ensure the country column exists
    if "country" not in mapping.columns:
        raise KeyError("The CSV file does not have a 'country' column.")

    # Update the ISO_country column without using apply
    if "ISO_country" not in mapping.columns:
        mapping["ISO_country"] = None  # Initialize the column if it doesn't exist

    for index, row in mapping.iterrows():
        mapping.at[index, "ISO_country"] = helper.get_iso_alpha_3(row["country"], row["ISO_country"])

    # Save the updated DataFrame back to the same CSV
    mapping.to_csv(common.get_configs("mapping"), index=False)

    logger.info("Mapping file updated with ISO codes.")

if common.get_configs("update_pop_country"):
    helper.update_population_in_csv(mapping)

if common.get_configs("update_continent"):
    # Update the continent column based on the country
    mapping['continent'] = mapping['country'].apply(helper.get_continent_from_country)

    # Save the updated CSV file
    mapping.to_csv(common.get_configs("mapping"), index=False)
    logger.info("Mapping file updated successfully with continents.")

if common.get_configs("update_mortality_rate"):
    helper.fill_traffic_mortality(mapping)

if common.get_configs("update_gini_value"):
    helper.fill_gini_data(mapping)

if common.get_configs("update_fps_list"):
    helper.update_csv_with_fps(mapping)

if common.get_configs("update_upload_date"):
    # Process the 'videos' column
    def extract_upload_dates(video_column):
        upload_dates = []
        for video_list in video_column:
            # Parse the video IDs from the string
            video_ids = video_list.strip('[]').split(',')
            video_ids = [vid.strip() for vid in video_ids]

            # Get upload dates for all video IDs in the list
            dates = [helper.get_upload_date(vid) for vid in video_ids]

            # Combine dates in the same format as requested
            upload_dates.append(f"[{','.join(date if date else 'None' for date in dates)}]")
        return upload_dates

    mapping['upload_date'] = extract_upload_dates(mapping['videos'])

    # Save the updated file
    mapping.to_csv(common.get_configs("mapping"), index=False)
    logger.info("Mapping file updated successfully with upload dates.")


for index, row in mapping.iterrows():
    video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
    start_times = ast.literal_eval(row["start_time"])
    end_times = ast.literal_eval(row["end_time"])
    time_of_day = ast.literal_eval(row["time_of_day"])
    iso_country = ast.literal_eval(row["ISO_country"])
    if pd.isna(row["fps_list"]) or row["fps_list"] == '[]':
        fps_values = [60 for _ in range(len(video_ids))]
    else:
        fps_values = ast.literal_eval(row["fps_list"])

    for vid_index, (vid, start_times_list, end_times_list, time_of_day_list) in enumerate(zip(
            video_ids, start_times, end_times, time_of_day)):
        for start_time, end_time, time_of_day_value in zip(start_times_list, end_times_list, time_of_day_list):
            file_name = f'{vid}_{start_time}.csv'
            file_path = os.path.join(common.get_configs("data"), file_name)
            print(common.get_configs("countries_analyse"))
            print(iso_country)
            # Check if countries is in the list to be analysed
            if countries_analyse and iso_country not in countries_analyse:
                pass
            # Check if the file exists
            if os.path.isfile(file_path):
                pass
            else:
                # Attempt to download the video
                result = helper.download_video_with_resolution(video_id=vid, output_path=output_path)
                # result = None
                if result:
                    video_file_path, video_title, resolution, fps = result
                    # Update the fps value for the current video
                    if len(fps_values) <= vid_index:
                        # Extend the list if the index doesn't exist yet
                        fps_values.extend([60 for _ in range(vid_index - len(fps_values) + 1)])  # type: ignore

                    # Update the specific FPS value for the current video and index
                    fps_values[vid_index] = fps  # type: ignore # Replace the list with the new FPS value

                    # Dynamically update the 'fps_list' column for the current row
                    mapping.at[index, 'fps_list'] = str(fps_values)

                    # Write the updated DataFrame back to the CSV file after every update
                    mapping.to_csv(common.get_configs("mapping"), index=False)
                    logger.info(f"Downloaded video: {video_file_path}")
                    helper.set_video_title(video_title)
                else:
                    # If download fails, check if the video already exists in the folder
                    video_file_path = os.path.join(output_path, f"{vid}.mp4")
                    if os.path.exists(video_file_path):
                        video_title = vid  # Assuming the video ID is the title for simplicity
                        logger.info(f"Video found: {video_file_path}")
                        helper.set_video_title(video_title)
                    else:
                        logger.error(f"Video {vid} not found and download failed. Skipping this video.")
                        continue

                input_video_path = video_file_path
                output_video_path = os.path.join(output_path, f"{video_title}_mod.mp4")

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

                if common.get_configs("prediction_mode"):
                    helper.prediction_mode()

                if common.get_configs("tracking_mode"):
                    if fps_values[vid_index]:  # Get the last FPS value
                        tracking_fps = fps_values[vid_index]
                        helper.tracking_mode(input_video_path, output_video_path, tracking_fps)
                    else:
                        logger.warning(f"FPS not found for video ID: {vid}. Skipping tracking mode.")

                    os.makedirs(data_folder, exist_ok=True)
                    old_file_path = os.path.join("runs", "detect", f"{vid}.csv")
                    new_file_path = os.path.join("runs", "detect", f"{vid}_{start_time}.csv")

                    os.rename(old_file_path, new_file_path)
                    # Construct the paths dynamically
                    source_file = os.path.join("runs", "detect", f"{vid}_{start_time}.csv")
                    predict_folder = os.path.join("runs", "detect")

                    # Move the file to the data_folder
                    shutil.move(source_file, data_folder)

                    if delete_runs_files:
                        shutil.rmtree(os.path.join("runs", "detect"))
                    else:
                        source_folder = os.path.join("runs", "detect")
                        destination_folder = os.path.join("runs", f"{video_title}_{resolution}_{datetime.now()}")

                        helper.rename_folder(source_folder, destination_folder)
                    counter += 1

                if delete_youtube_video:
                    os.remove(input_video_path)
