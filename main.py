# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import shutil
import os
from datetime import datetime
from helper_script import Youtube_Helper
import pandas as pd
from custom_logger import CustomLogger
from logmod import logs
import ast
import common
from tqdm import tqdm
import time


logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger
helper = Youtube_Helper()

while True:  # run this script loop forever
    # Load the config file
    mapping = pd.read_csv(common.get_configs("mapping"))
    video_paths = common.get_configs("videos")  # folders with videos
    output_path = common.get_configs("videos")[-1]  # use the last folder with videos to download
    delete_runs_files = common.get_configs("delete_runs_files")
    delete_youtube_video = common.get_configs("delete_youtube_video")
    data_folder = common.get_configs("data")
    countries_analyse = common.get_configs("countries_analyse")

    if common.get_configs("check_missing_mapping"):
        helper.check_missing_mapping(mapping)

    if common.get_configs("update_ISO_code"):
        # Ensure the country column exists
        if "country" not in mapping.columns:
            raise KeyError("The CSV file does not have a 'country' column.")

        # Update the iso3 column without using apply
        if "iso3" not in mapping.columns:
            mapping["iso3"] = None  # Initialize the column if it doesn't exist

        for index, row in mapping.iterrows():
            mapping.at[index, "iso3"] = helper.get_iso_alpha_3(row["country"], row["iso3"])

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

    # Delete the runs folder (if it exists)
    helper.delete_folder(folder_path="runs")

    # create required directories
    os.makedirs(data_folder, exist_ok=True)

    # Go over rows. Add progress bar.
    for index, row in tqdm(mapping.iterrows(), total=mapping.shape[0]):
        video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
        start_times = ast.literal_eval(row["start_time"])
        end_times = ast.literal_eval(row["end_time"])
        time_of_day = ast.literal_eval(row["time_of_day"])
        iso3 = str(row["iso3"])
        # Check if countries is in the list to be analysed
        if countries_analyse and iso3 not in countries_analyse:
            continue
        city = str(row["city"])
        state = str(row["state"])
        country = str(row["country"])
        logger.info(f"Processing videos for city={city}, state={state}, country={country}.")
        if pd.isna(row["fps_list"]) or row["fps_list"] == '[]':
            fps_values = [0 for _ in range(len(video_ids))]
        else:
            fps_values = ast.literal_eval(row["fps_list"])

        for vid_index, (vid, start_times_list, end_times_list, time_of_day_list) in enumerate(zip(
                video_ids, start_times, end_times, time_of_day)):

            # Define a base video file path for the downloaded original video
            base_video_path = os.path.join(output_path, f"{vid}.mp4")

            video_fps = 0.0  # store detected FPS value

            # If the base video does not exist, attempt to download it
            if not any(os.path.exists(os.path.join(path, f"{vid}.mp4")) for path in video_paths):
                result = helper.download_video_with_resolution(vid=vid, output_path=output_path)
                if result:
                    video_file_path, video_title, resolution, video_fps = result
                    # Set the base video path to the downloaded video
                    base_video_path = video_file_path

                    if common.get_configs("update_fps_list"):
                        # Update the FPS information for the current video
                        if len(fps_values) <= vid_index:
                            fps_values.extend([60] * (vid_index - len(fps_values) + 1))
                        fps_values[vid_index] = video_fps  # type: ignore

                        # Update the DataFrame mapping and write to CSV
                        mapping.at[index, 'fps_list'] = str(fps_values)
                        mapping.to_csv(common.get_configs("mapping"), index=False)

                    logger.info(f"{vid}: downloaded to {video_file_path}.")
                    helper.set_video_title(video_title)

                    # Optionally compress the video if required
                    if common.get_configs("compress_youtube_video"):
                        helper.compress_video(base_video_path)
                else:
                    # If download fails, check if the video already exists
                    if os.path.exists(base_video_path):
                        video_title = vid  # or any fallback title
                        logger.info(f"{vid}: download failed, but video found: {base_video_path}.")
                        helper.set_video_title(video_title)
                    else:
                        logger.error(f"{vid}: video not found and download failed. Skipping.")
                        continue
            else:
                logger.info(f"{vid}: using already downloaded video.")
                # find the first folder where the file exists
                existing_folder = next((path for path in video_paths if os.path.exists(os.path.join(path, f"{vid}.mp4"))), None)  # noqa: E501
                # if the file exists, use that folder; otherwise, default to the last folder
                existing_path = existing_folder if existing_folder else video_paths[-1]
                base_video_path = os.path.join(existing_path, f"{vid}.mp4")
                video_title = vid  # or any fallback title
                helper.set_video_title(video_title)
                video_fps = helper.get_video_fps(base_video_path)  # try to get FPS value of existing file

            for start_time, end_time, time_of_day_value in zip(start_times_list, end_times_list, time_of_day_list):
                # Construct a unique file name for the trimmed segment
                trimmed_file_path = os.path.join(data_folder, f'{vid}_{start_time}.csv')

                # If the YOLO output file already exists, skip processing for this segment
                if os.path.isfile(trimmed_file_path) and (common.get_configs("prediction_mode") or common.get_configs("tracking_mode")):  # noqa: E501
                    logger.info(f"{vid}: YOLO file {vid}_{start_time}.csv exists. Skipping segment.")
                    continue

                # Define a temporary path for the trimmed video segment
                trimmed_video_path = os.path.join(output_path, f"{video_title}_mod.mp4")

                if start_time is None and end_time is None:
                    logger.info(f"{vid}: no trimming required for this video.")
                elif common.get_configs("prediction_mode") or common.get_configs("tracking_mode"):
                    # trim only if needed
                    logger.info(f"{vid}: trimming in progress for segment {start_time}-{end_time}s.")
                    # Adjust end_time if needed (e.g., to account for missing frames)
                    end_time_adj = end_time - 1
                    helper.trim_video(base_video_path, trimmed_video_path, start_time, end_time_adj)

                    logger.info(f"{vid}: trimming completed for segment {start_time}-{end_time}s.")

                if common.get_configs("prediction_mode"):
                    helper.prediction_mode()

                # Tracking mode: process the trimmed segment
                if common.get_configs("tracking_mode"):
                    if video_fps > 0:  # type: ignore
                        logger.info(f"{vid}: YOLO analysis in progress for segment {start_time}-{end_time}s with FPS from file {video_fps}.")  # noqa: E501
                        helper.tracking_mode(trimmed_video_path, trimmed_video_path, video_fps)  # type: ignore
                    elif fps_values[vid_index] > 0:
                        logger.info(f"{vid}: YOLO analysis in progress for segment {start_time}-{end_time}s with FPS from mapping {fps_values[vid_index]}.")  # noqa: E501
                        helper.tracking_mode(trimmed_video_path, trimmed_video_path, fps_values[vid_index])
                    else:
                        logger.warning(f"{vid}: FPS value is {video_fps}. Skipping tracking mode.")

                    # Move and rename the generated CSV file from tracking mode
                    old_file_path = os.path.join("runs", "detect", f"{vid}.csv")
                    new_file_path = os.path.join("runs", "detect", f"{vid}_{start_time}.csv")
                    if os.path.exists(old_file_path):
                        os.rename(old_file_path, new_file_path)
                    else:
                        logger.error(f"{vid}: error:{old_file_path} does not exist.")

                    # Move the CSV file to the desired folder
                    if os.path.exists(new_file_path):
                        shutil.move(new_file_path, data_folder)
                    else:
                        logger.error(f"{vid}: error: {new_file_path} does not exist.")

                    if delete_runs_files:
                        shutil.rmtree(os.path.join("runs", "detect"))
                    else:
                        source_folder = os.path.join("runs", "detect")
                        destination_folder = os.path.join("runs", f"{video_title}_{resolution}_{datetime.now()}")
                        helper.rename_folder(source_folder, destination_folder)
                # remove trimmed file
                if common.get_configs("prediction_mode") or common.get_configs("tracking_mode"):
                    os.remove(trimmed_video_path)

            # Optionally delete the original video after processing if needed
            if delete_youtube_video:
                os.remove(base_video_path)

    # Pause the file for sleep_sec seconds before doing analysis again
    sleep_sec = common.get_configs("sleep_sec")
    logger.info(f"Sleeping for {sleep_sec} before attempting to go over mapping again.")
    time.sleep(sleep_sec)
