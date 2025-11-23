# interactive_video_analyzer.py

import os
import pandas as pd
from dotenv import load_dotenv
from helper_script import Youtube_Helper
import re

# --- Load FTP credentials from .env ---
load_dotenv()
ftp_server = os.getenv("FTP_SERVER")
ftp_username = os.getenv("FTP_USERNAME")
ftp_password = os.getenv("FTP_PASSWORD")

# --- Folders ---
videos_folder = "./videos"
output_folder = "./outputs"
os.makedirs(videos_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# --- Load and clean mapping file ---
mapping_file = "./mapping.csv"
if not os.path.exists(mapping_file):
    raise FileNotFoundError("mapping.csv not found! Place it in project folder.")

mapping_df = pd.read_csv(mapping_file)

# ✅ Normalize the 'videos' column to handle list-like strings
if "videos" not in mapping_df.columns:
    raise ValueError("mapping.csv must contain a 'videos' column.")

# Remove brackets, quotes, and spaces
mapping_df["videos"] = (
    mapping_df["videos"]
    .astype(str)
    .str.replace(r"[\[\]\s\"']", "", regex=True)
)

# Split multiple IDs in a single cell and expand them into separate rows
mapping_df = mapping_df.assign(videos=mapping_df["videos"].str.split(",")).explode("videos").reset_index(drop=True)

# Clean start_time and end_time if present
if "start_time" in mapping_df.columns:
    mapping_df["start_time"] = mapping_df["start_time"].astype(str).str.replace(r"[\[\]\s]", "", regex=True)
if "end_time" in mapping_df.columns:
    mapping_df["end_time"] = mapping_df["end_time"].astype(str).str.replace(r"[\[\]\s]", "", regex=True)

# --- Initialize helper ---
helper = Youtube_Helper()

print("\n🎯 Interactive Video Analyzer")
print("---------------------------------------------------")
print("Enter a video ID from mapping.csv (or type 'exit' to quit)\n")

# --- Main Loop ---
while True:
    video_id = input("Enter video ID: ").strip()

    if video_id.lower() in ["exit", "quit", "q"]:
        print("\n👋 Exiting video analyzer. Goodbye!")
        break

    # --- Lookup video ID in mapping ---
    match_row = mapping_df[mapping_df["videos"].str.lower() == video_id.lower()]

    if match_row.empty:
        print(f"⚠️ Video ID {video_id} not found in mapping.csv.")
        continue

    row = match_row.iloc[0]
    start_time = None
    end_time = None
    city = row.get("city", "")

    # Handle optional start/end time columns
    try:
        start_time = float(row["start_time"]) if row["start_time"] not in ["", "nan", "None"] else None
        end_time = float(row["end_time"]) if row["end_time"] not in ["", "nan", "None"] else None
    except Exception:
        start_time, end_time = None, None

    print(f"\n🎬 Found {video_id} ({city})")
    print(f"Start time: {start_time or 'N/A'}s | End time: {end_time or 'N/A'}s")

    try:
        # Step 1: Download video (FTP first)
        print("📥 Downloading video from FTP...")
        result = helper.download_videos_from_ftp(
            filename=video_id,
            base_url=ftp_server,
            out_dir=videos_folder,
            username=ftp_username,
            password=ftp_password
        )

        # Step 2: Fallback to YouTube if FTP fails
        if not result:
            print("⚠️ FTP failed — trying YouTube...")
            result = helper.download_video_with_resolution(vid=video_id, output_path=videos_folder)

        if not result:
            print("❌ Could not download video. Try another ID.")
            continue

        video_path, video_title, resolution, fps = result
        print(f"✅ Downloaded: {video_path} ({resolution}, {fps} FPS)")

        # Step 3: Trim video if time range provided
        trimmed_path = video_path
        if start_time is not None and end_time is not None and end_time > start_time:
            trimmed_path = os.path.join(videos_folder, f"{video_id}_trimmed.mp4")
            helper.trim_video(video_path, trimmed_path, start_time, end_time)
            print(f"✂️ Trimmed segment saved at {trimmed_path}")

        # Step 4: Run YOLO detection/tracking
        print("🔍 Running object detection/tracking...")
        fps_val = helper.get_video_fps(trimmed_path) or 30
        helper.tracking_mode(
            input_video_path=trimmed_path,
            output_video_path=output_folder,
            video_title=os.path.basename(trimmed_path),
            video_fps=int(fps_val),
            seg_mode=False,   # Bounding-box mode
            bbox_mode=True,   # Detection active
            flag=1            # Save annotated video
        )

        print(f"✅ Detection complete for {video_id}")
        print(f"🎞️ Output saved in: {output_folder}\n")

        # Step 5: Continue or exit
        next_action = input("Analyze another video? (y/n): ").strip().lower()
        if next_action not in ["y", "yes"]:
            print("👋 Exiting system. Goodbye!")
            break

    except Exception as e:
        print(f"⚠️ Error while processing {video_id}: {e}")
