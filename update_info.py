# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import ast
from pytubefix import YouTube
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import csv
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

metadata_file = "mapping_metadata.csv"
csv_headers = [
    "id", "video", "title", "upload_date", "channel", "views", "description",
    "chapters", "segments", "date_updated"
]


def safe_parse(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            if val.startswith('[') and val.endswith(']'):
                inner = val[1:-1]
                return [item.strip().strip("'\"") for item in inner.split(',') if item.strip()]
    return []


def clean_text(val):
    if isinstance(val, list):
        return str(val)
    if pd.isna(val):
        return ""
    return str(val).replace('\n', ' ').replace('\r', ' ').strip()


def get_video_info(video_id):
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        title = yt.title
        description = yt.description
        channel = yt.channel_id
        views = yt.views
        upload_date = yt.publish_date.strftime('%d%m%Y') if yt.publish_date else None

        chapters = []
        if yt.chapters:
            for c in yt.chapters:
                try:
                    chapters.append({
                        "title": c.title,
                        "timestamp": str(timedelta(seconds=c.start_seconds))
                    })
                except Exception as e:
                    logger.error(f"⚠️ failed to extract chapter info in {video_id}: {e}")

        safe_title = str(title).replace("{", "{{").replace("}", "}}")
        safe_channel = str(channel).replace("{", "{{").replace("}", "}}")

        logger.info(
            f"✅ fetched: {video_id} | title: {safe_title} | upload: {upload_date} | channel: {safe_channel} "
            f"| views: {views}"
        )

        return {
            "video": video_id,
            "title": clean_text(title),
            "upload_date": upload_date,
            "channel": clean_text(channel),
            "views": views,
            "description": clean_text(description),
            "chapters": clean_text(chapters),
        }
    except Exception as e:
        logger.info(f"❌ failed to fetch {video_id}: {e}")
        return {
            "video": video_id,
            "title": "",
            "upload_date": "",
            "channel": "",
            "views": "",
            "description": "",
            "chapters": "",
        }


if __name__ == "__main__":
    df = pd.read_csv(common.get_configs("mapping"))

    # collect all video IDs from mapping file
    all_video_ids = []
    for row in df['videos']:
        vids = safe_parse(row)
        all_video_ids.extend(vids)

    video_count = pd.Series(all_video_ids).value_counts().to_dict()
    unique_video_ids = list(video_count.keys())

    # handle metadata file
    if os.path.exists(metadata_file):
        existing_df = pd.read_csv(metadata_file)
        existing_ids = set(existing_df['video'].tolist())
        last_id = existing_df['id'].max() if not existing_df.empty else 0
        logger.info(f"🗃️ found {len(existing_ids)} videos in existing metadata")

        # find failed entries
        failed_mask = (
            existing_df['title'].isna() |
            (existing_df['title'].astype(str).str.strip() == "")
        )
        failed_ids = set(existing_df.loc[failed_mask, 'video'].tolist())
        logger.info(f"🔁 found {len(failed_ids)} videos with failed metadata fetch")
    else:
        with open(metadata_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            writer.writerow(csv_headers)
        existing_df = pd.DataFrame(columns=csv_headers)
        existing_ids = set()
        failed_ids = set()
        last_id = 0
        logger.info("📁 created new metadata file")

    # decide which videos to fetch
    new_videos = [vid for vid in unique_video_ids if vid not in existing_ids]
    retry_videos = [vid for vid in failed_ids if vid in unique_video_ids]
    videos_to_fetch = new_videos + retry_videos

    logger.info(f"🔍 fetching data for {len(videos_to_fetch)} videos "
                f"({len(new_videos)} new, {len(retry_videos)} retries)")

    now = datetime.now().strftime('%d%m%Y')
    current_id = last_id

    for vid in tqdm(videos_to_fetch):
        info = get_video_info(vid)
        info["segments"] = video_count.get(vid, 1)
        info["date_updated"] = now

        # existing id or new one
        if vid in existing_ids:
            existing_id = existing_df.loc[existing_df['video'] == vid, 'id'].values[0]
            info["id"] = existing_id
            # update row in DataFrame
            existing_df.loc[existing_df['video'] == vid, list(info.keys())] = list(info.values())
        else:
            current_id += 1
            info["id"] = current_id
            existing_df = pd.concat([existing_df, pd.DataFrame([info])], ignore_index=True)

        # write to CSV immediately (safe append)
        existing_df.to_csv(metadata_file, index=False)
        logger.info(f"💾 updated metadata for {vid}")

    logger.info("✅ all videos processed and metadata file updated.")