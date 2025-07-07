# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import csv
import os
import subprocess
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

# config
video_dirs = common.get_configs("videos")  # list of video folders
mapping_file = common.get_configs("mapping")
output_file = "mapping_updated.csv"


def get_fps(video_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        rate = result.stdout.strip()
        if "/" in rate:
            num, denom = map(int, rate.split("/"))
            return round(num / denom)
        return round(float(rate))
    except Exception as e:
        logger.warning(f"could not extract fps from {video_path}: {e}")
        return None


def find_video_path(video_id):
    for folder in video_dirs:
        path = os.path.join(folder, f"{video_id}.mp4")
        if os.path.isfile(path):
            return path
    return None


def parse_video_list(raw):
    return [v.strip() for v in raw.strip()[1:-1].split(",") if v.strip()]


def write_updated_csv(rows, header):
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


updated_rows = []
with open(mapping_file, "r", newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames

    for row in reader:
        video_ids = parse_video_list(row["videos"])
        fps_values = []

        for vid in video_ids:
            path = find_video_path(vid)
            if path:
                fps = get_fps(path)
                if fps is not None:
                    fps_values.append(str(fps))
                else:
                    fps_values.append("0")
            else:
                logger.warning(f"video not found: {vid}")
                fps_values.append("0")

        row["fps_list"] = f"[{','.join(fps_values)}]"
        updated_rows.append(row)

write_updated_csv(updated_rows, header)
logger.info(f"Updated mapping file written to {output_file}")
