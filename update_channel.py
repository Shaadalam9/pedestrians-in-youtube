# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import ast
from pytubefix import YouTube
from tqdm import tqdm
import common


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


def get_channel_id(video_id):
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        yt_channel = yt.channel_id
        print(f"✅ channel for https://www.youtube.com/watch?v={video_id} → {yt_channel}")
        return yt_channel
    except Exception as e:
        print(f"❌ error fetching channel ID for {video_id}: {e}")
        return None


if __name__ == "__main__":
    # Load the CSV
    df = pd.read_csv(common.get_configs("mapping"))

    # Ensure 'channel' column exists
    if 'channel' not in df.columns:
        df['channel'] = ''

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_ids = safe_parse(row['videos'])
        channel_ids = safe_parse(row['channel'])

        # make sure the list lengths match
        if len(channel_ids) != len(video_ids):
            channel_ids = [None] * len(video_ids)

        updated_channels = []
        for vid, ch in zip(video_ids, channel_ids):
            if ch in [None, 'None', '', 'nan']:
                ch_id = get_channel_id(vid)
                updated_channels.append(ch_id if ch_id else None)
            else:
                updated_channels.append(ch)

        # store clean list format like [UCabc,UCxyz]
        df.at[idx, 'channel'] = f"[{','.join([c for c in updated_channels if c])}]"

    # Save updated CSV
    output_file = 'mapping_with_channels.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ updated data with clean channel list saved to {output_file}")
