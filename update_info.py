# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import ast
from pytubefix import YouTube
from tqdm import tqdm
from datetime import datetime
import os
import common
from datetime import timedelta


def safe_parse(val):
    if pd.isna(val) or not isinstance(val, str):
        return []
    val = val.strip()
    try:
        return ast.literal_eval(val)
    except Exception:
        pass
    if val.startswith('[') and val.endswith(']'):
        inner = val[1:-1]
        return [item.strip().strip("'\"") for item in inner.split(',') if item.strip()]
    return []


def get_video_info(video_id):
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        title = yt.title
        description = yt.description
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
                    print(f"⚠️ failed to extract chapter info in {video_id}: {e}")
        print(f"✅ fetched: {video_id} | title: {title} | upload: {upload_date} | chapters: {chapters}")
        return {
            "video": video_id,
            "title": title,
            "description": description,
            "chapters": chapters,
            "upload_date": upload_date,
        }
    except Exception as e:
        print(f"❌ failed to fetch {video_id}: {e}")
        return {
            "video": video_id,
            "title": None,
            "description": None,
            "chapters": [],
            "upload_date": None,
        }


if __name__ == "__main__":
    # load mapping csv
    df = pd.read_csv(common.get_configs("mapping"))

    # collect all video ids from mapping
    all_video_ids = []
    for row in df['videos']:
        vids = safe_parse(row)
        all_video_ids.extend(vids)

    video_count = pd.Series(all_video_ids).value_counts().to_dict()
    unique_video_ids = list(video_count.keys())

    # load existing metadata if present
    metadata_file = "mapping_metadata.csv"
    if os.path.exists(metadata_file):
        existing_df = pd.read_csv(metadata_file, converters={"chapters": ast.literal_eval})
        existing_ids = set(existing_df['video'].tolist())
        print(f"🗃️ found {len(existing_ids)} videos in existing metadata")
    else:
        existing_df = pd.DataFrame()
        existing_ids = set()

    # only fetch missing ones
    videos_to_fetch = [vid for vid in unique_video_ids if vid not in existing_ids]
    print(f"🔍 fetching data for {len(videos_to_fetch)} new videos")

    # get today's date
    now = datetime.now().strftime('%d%m%Y')

    # fetch new metadata
    new_records = []
    for vid in tqdm(videos_to_fetch):
        info = get_video_info(vid)
        info["count"] = video_count.get(vid, 1)
        info["date_updated"] = now
        new_records.append(info)

    # merge and save
    if new_records:
        new_df = pd.DataFrame(new_records)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"💾 added {len(new_records)} new records")
    else:
        final_df = existing_df
        print("✅ no new videos to update")

    final_df.to_csv(metadata_file, index=False)
    print(f"✅ updated metadata saved to {metadata_file}")
