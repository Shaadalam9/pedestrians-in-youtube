# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import pandas as pd
import common

# get the folder where downloaded videos are stored
video_folder = common.get_configs('videos')

# get a list of already downloaded video files (assuming mp4 format)
downloaded_videos = {f.split('.')[0] for f in os.listdir(video_folder) if f.endswith('.mp4')}

# load mapping.csv
mapping_file = 'mapping.csv'
df = pd.read_csv(mapping_file)


# function to check if a video id is already downloaded
def filter_videos(video_list):
    if not isinstance(video_list, str) or not video_list.strip():
        return []
    try:
        # remove surrounding quotes and brackets
        video_list = video_list.strip().strip('"[]')
        # split by comma and remove spaces
        video_ids = [vid.strip() for vid in video_list.split(',')]
        return [vid for vid in video_ids if vid and vid not in downloaded_videos]
    except Exception:
        return []  # return empty list if parsing fails


# filter rows where there are videos that haven't been downloaded
df['filtered_videos'] = df['videos'].apply(filter_videos)
df_filtered = df[df['filtered_videos'].map(len) > 0]

df_filtered = df_filtered.drop(columns=['videos']).rename(columns={'filtered_videos': 'videos'})

# save the filtered subset
subset_file = 'mapping_queue.csv'
df_filtered.to_csv(subset_file, index=False)

print(f'filtered mapping saved to {subset_file}')
