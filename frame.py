import os
import cv2

# Set these values directly in the code
VIDEO_PATH = "videos/qOx5CwCrN9k.mp4"
TIME_IN_SECONDS = 897
OUTPUT_IMAGE = "screenshots/seoul.jpg"


def extract_frame(video_path: str, time_in_seconds: float, output_image: str) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if time_in_seconds < 0:
        raise ValueError("Time in seconds must be 0 or greater")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError("Could not read video metadata")

    duration = frame_count / fps
    if time_in_seconds > duration:
        cap.release()
        raise ValueError(
            f"Requested time {time_in_seconds}s is beyond video duration {duration:.2f}s"
        )

    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise RuntimeError(f"Could not extract frame at {time_in_seconds}s")

    saved = cv2.imwrite(output_image, frame)
    if not saved:
        raise RuntimeError(f"Could not save frame to: {output_image}")

    print(f"Frame saved to: {output_image}")


extract_frame(VIDEO_PATH, TIME_IN_SECONDS, OUTPUT_IMAGE)