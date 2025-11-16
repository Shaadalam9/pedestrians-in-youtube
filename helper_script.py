# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import re
import time
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from ultralytics import YOLO
from collections import defaultdict
from typing import Optional, Set, Any
import shutil
import numpy as np
import pandas as pd
import world_bank_data as wb
import yt_dlp
import pycountry
from pycountry_convert import country_name_to_country_alpha2, country_alpha2_to_continent_code
from custom_logger import CustomLogger
import common
import torch
import subprocess
import sys
import logging
from tqdm import tqdm
import datetime
import json
import yaml
import pathlib
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


logger = CustomLogger(__name__)  # use custom logger
logging.getLogger("ultralytics").setLevel(logging.ERROR)  # Show only errors

# Consts
LINE_THICKNESS = 1
RENDER = False
SHOW_LABELS = False
SHOW_CONF = False

# logging of attempts to upgrade packages
UPGRADE_LOG_FILE = "upgrade_log.json"


class Youtube_Helper:
    """
    A helper class for managing YouTube video downloads, processing, and analytics.

    Features:
        - Downloads videos via pytube or yt_dlp with resolution preference.
        - Handles video compression, trimming, and FPS extraction.
        - Applies object detection and tracking using YOLO models.
        - Updates and maintains CSV datasets with video metadata.
        - Interfaces with World Bank data to supplement mapping files.

    Attributes:
        model (str): Path or identifier for the YOLO model to use.
        resolution (str): Target resolution for downloaded videos.
        video_title (str): Title of the currently processed video.
    """

    def __init__(self, video_title=None):
        """
        Initialises a new instance of the class.

        Parameters:
            video_title (str, optional): The title of the video. Defaults to None.

        Instance Variables:
            self.model (str): The model configuration loaded from common.get_configs("model").
            self.resolution (str): The video resolution. Initialised as None and set later when needed.
            self.video_title (str): The title of the video.
        """
        self.tracking_model = common.get_configs("tracking_model")
        self.segment_model = common.get_configs("segment_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.seg_tracker = common.get_configs("seg_tracker")
        self.resolution = None
        self.mapping = pd.read_csv(common.get_configs("mapping"))
        self.confidence = 0.0
        self.display_frame_tracking = common.get_configs("display_frame_tracking")
        self.display_frame_segmentation = common.get_configs("display_frame_segmentation")
        self.output_path = common.get_configs("videos")
        self.save_annoted_img = common.get_configs("save_annoted_img")
        self.save_tracked_img = common.get_configs("save_tracked_img")
        self.delete_labels = common.get_configs("delete_labels")
        self.delete_frames = common.get_configs("delete_frames")
        self.update_package = common.get_configs("update_package")
        self.need_authentication = common.get_configs("need_authentication")
        self.client = common.get_configs("client")

    def set_video_title(self, title):
        """
        Sets the video title for the instance.

        Parameters:
            title (str): The new title for the video.
        """
        self.video_title = title

    def rename_folder(self, old_name, new_name):
        """
        Renames a folder from old_name to new_name.

        Parameters:
            old_name (str): The current name (or path) of the folder.
            new_name (str): The new name (or path) to assign to the folder.

        Error Handling:
            - Logs an error if the folder with old_name is not found.
            - Logs an error if a folder with new_name already exists.
        """
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            logger.error(f"Error: Folder '{old_name}' not found.")
        except FileExistsError:
            logger.error(f"Error: Folder '{new_name}' already exists.")

    def load_upgrade_log(self):
        """
        Load package upgrade attempt log from file.

        Returns:
            dict: Dictionary with package names and last upgrade date.
        """
        if not os.path.exists(UPGRADE_LOG_FILE):
            return {}
        try:
            with open(UPGRADE_LOG_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return {}

    def save_upgrade_log(self, log_data):
        """
        Save package upgrade log to a JSON file.

        Parameters:
            log_data (dict): Dictionary of package upgrade dates.
        """
        with open(UPGRADE_LOG_FILE, "w") as file:
            json.dump(log_data, file)

    def was_upgraded_today(self, package_name):
        """
        Check whether the given package was already upgraded today.

        Parameters:
            package_name (str): Name of the package.

        Returns:
            bool: True if upgraded today, False otherwise.
        """
        log_data = self.load_upgrade_log()
        today = datetime.date.today().isoformat()
        return log_data.get(package_name) == today

    def mark_as_upgraded(self, package_name):
        """
        Mark a package as upgraded by saving today's date in the log.

        Parameters:
            package_name (str): Name of the package.
        """
        log_data = self.load_upgrade_log()
        log_data[package_name] = datetime.date.today().isoformat()
        self.save_upgrade_log(log_data)

    def upgrade_package_if_needed(self, package_name):
        """
        Upgrades a given Python package using pip if it hasn't been attempted today.

        Parameters:
            package_name (str): The name of the package to upgrade.
        """
        if self.was_upgraded_today(package_name):
            logging.debug(f"{package_name} upgrade already attempted today. Skipping.")
            return

        try:
            logging.info(f"Upgrading {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            logging.info(f"{package_name} upgraded successfully.")
            self.mark_as_upgraded(package_name)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to upgrade {package_name}: {e}")
            self.mark_as_upgraded(package_name)  # still log it to avoid retrying

    @staticmethod
    def _wait_for_stable_file(src, checks=2, interval=0.5, timeout=30):
        """
        Wait until `src` exists and its size is unchanged for `checks` consecutive checks.
        Returns True if stable within timeout, else False.
        """
        deadline = time.time() + timeout
        last = -1
        stable = 0
        while time.time() < deadline:
            if os.path.isfile(src):
                try:
                    size = os.stat(src).st_size
                except OSError:
                    size = -1
                if size == last:
                    stable += 1
                    if stable >= checks:
                        return True
                else:
                    stable = 0
                last = size
            time.sleep(interval)
        return False

    def copy_video_safe(self, base_video_path, internal_ssd, vid, max_attempts=5, backoff=0.6):
        """
        Copies base_video_path -> os.path.join(internal_ssd, f"{vid}.mp4")
        - Ensures dest dir exists
        - Waits for source file to be stable
        - Copies to temp, then atomic replace
        - Verifies size
        - Retries on transient OS errors
        """
        if not vid or str(vid).strip() == "":
            raise ValueError("vid must be a non-empty string")

        dest_dir = internal_ssd
        dest = os.path.join(dest_dir, f"{vid}.mp4")
        tmp = dest + ".tmp"

        # Ensure destination directory
        os.makedirs(dest_dir, exist_ok=True)

        # Avoid copying onto itself (only works if both exist)
        try:
            if os.path.exists(dest) and os.path.exists(base_video_path) and os.path.samefile(base_video_path, dest):
                return dest
        except OSError:
            # samefile can raise if either path doesn't fully exist yet; ignore
            pass

        # Wait for source to appear and stabilize
        if not Youtube_Helper._wait_for_stable_file(base_video_path):
            raise FileNotFoundError(f"Source never became available or stayed unstable: {base_video_path}")

        # Cache expected size for verification
        try:
            src_size = os.stat(base_video_path).st_size
        except OSError as e:
            raise FileNotFoundError(f"Cannot stat source: {base_video_path}: {e!r}")

        attempt = 0
        while True:
            attempt += 1
            try:
                # Copy to temp
                shutil.copy2(base_video_path, tmp)

                # Best-effort flush of temp file contents
                try:
                    with open(tmp, "rb") as f:
                        os.fsync(f.fileno())
                except (OSError, AttributeError):
                    pass

                # Atomic replace final
                os.replace(tmp, dest)

                # Verify size
                if os.stat(dest).st_size != src_size:
                    raise OSError(f"Size mismatch after copy: src={src_size}, dst={os.stat(dest).st_size}")

                return dest

            except (OSError, IOError):
                # Clean up temp if present
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass

                if attempt >= max_attempts:
                    raise

                time.sleep(backoff * attempt)

    def download_videos_from_ftp(self, filename: str, base_url: Optional[str] = None, out_dir: str = ".",
                                 username: Optional[str] = None, password: Optional[str] = None,
                                 token: Optional[str] = None, timeout: int = 20, debug: bool = True,
                                 max_pages: int = 500,) -> Optional[tuple[str, str, str, float]]:

        """
        Search and download a specific `.mp4` file from a multi-directory
        FastAPI-based HTTP file server (e.g., files.mobility-squad.com).

        This function attempts direct download from known `/files/` paths
        (tue1/tue2/tue3), and if not found, recursively crawls the `/browse`
        pages to locate the video file. Progress is shown with `tqdm`.

        Args:
            filename (str): Target file name (with or without `.mp4` extension).
            base_url (str, optional): Base URL of the file server. Must include protocol,
                e.g. `"https://files.mobility-squad.com/"`.
            out_dir (str, optional): Local output directory to save the video.
                Defaults to current directory `"."`.
            username (str, optional): Username for HTTP Basic Auth.
            password (str, optional): Password for HTTP Basic Auth.
            token (str, optional): Token string for token-based authentication.
                Sent as a query parameter `?token=...`.
            timeout (int, optional): Request timeout in seconds. Default is 20.
            max_pages (int, optional): Safety limit for crawl depth/pages. Default is 500.

        Returns:
            Optional[Tuple[str, str, str, float]]:
                Returns a tuple `(local_path, filename, resolution_label, fps)`
                if the download succeeds, or `None` if the file is not found or
                download fails.

        Logging:
            - `logger.info`: start, success summaries.
            - `logger.debug`: HTTP requests, crawl steps, file matches.
            - `logger.warning`: non-fatal issues (metadata failures, skipped pages).
            - `logger.error`: fatal errors (network/IO exceptions).

        Example:
            ```python
            result = self.download_videos_from_http_fileserver(
                filename="3ai7SUaPoHM",
                base_url="https://files.mobility-squad.com/",
                out_dir="./downloads",
                username="mobility",
                password="your_password"
            )
            if result:
                path, name, res, fps = result
                print(f"Downloaded {name} ({res}, {fps} fps) to {path}")
            else:
                print("File not found or failed.")
            ```
        """

        # -------------------- Input Preparation --------------------
        if not base_url:
            logger.error("Base URL is missing.")
            return None

        base = base_url if base_url.endswith("/") else base_url + "/"
        if username == "":
            username = None
        if password == "":
            password = None

        filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
        filename_lower = filename_with_ext.lower()
        aliases = ["tue1", "tue2", "tue3", "tue4"]
        req_params = {"token": token} if token else None

        logger.info(f"Starting download for '{filename_with_ext}'")
        logger.debug(f"Base URL: {base} | Auth: {'Basic' if username and password else 'None'} | Token: {'Yes' if token else 'No'}")  # noqa: E501

        # ---------- Session ----------
        with requests.Session() as session:
            if username and password:
                session.auth = (username, password)
            session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

            def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
                """GET with logging and safe error handling."""
                try:
                    r = session.get(url, timeout=timeout, params=req_params, stream=stream)
                    logger.debug(f"GET {url} -> {r.status_code}")
                    if r.status_code == 401:
                        logger.error(f"Authentication failed for {url}")
                    r.raise_for_status()
                    return r
                except requests.RequestException as e:
                    logger.warning(f"Request failed [{url}]: {e}")
                    return None

            # ---------- 1. Try direct /files paths ----------
            for alias in aliases:
                direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
                logger.debug(f"Trying direct URL: {direct_url}")
                r = fetch(direct_url, stream=True)
                if r is None:
                    continue

                logger.info(f"Found file via direct URL: {direct_url}")
                content_len = int(r.headers.get("content-length", 0))
                logger.debug(f"Content-Length: {content_len or 'unknown'} bytes")

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                # Avoid overwriting
                if os.path.exists(local_path):
                    stem, suf = os.path.splitext(local_path)
                    i = 1
                    while os.path.exists(f"{stem} ({i}){suf}"):
                        i += 1
                    local_path = f"{stem} ({i}){suf}"
                    logger.warning(f"File exists, saving as: {local_path}")

                # ---------- Download ----------
                try:
                    total = content_len or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading from ftp: {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Download complete: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download failed for {filename_with_ext}: {e}")
                    return None

                # ---------- Metadata ----------
                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Youtube_Helper.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata extracted: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                logger.info(f"✅ Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
                return local_path, filename, resolution, fps

            # ---------- 2. Crawl /browse fallback ----------
            visited: Set[str] = set()

            def is_dir_link(href: str) -> bool:
                return href.startswith("/v/") and "/browse" in href

            def is_file_link(href: str) -> bool:
                return "/files/" in href

            def crawl(start_url: str) -> Optional[str]:
                """Recursively traverse /browse pages."""
                stack = [start_url]
                pages_seen = 0
                while stack:
                    url = stack.pop()
                    if url in visited:
                        continue
                    visited.add(url)
                    pages_seen += 1
                    if pages_seen > max_pages:
                        logger.warning(f"Crawl aborted after {max_pages} pages.")
                        return None

                    resp = fetch(url)
                    if resp is None:
                        continue

                    try:
                        soup = BeautifulSoup(resp.text, "html.parser")
                    except Exception as e:
                        logger.warning(f"HTML parse failed at {url}: {e}")
                        continue

                    for a in soup.find_all("a"):
                        href = (a.get("href") or "").strip()
                        if not href:
                            continue
                        full = urljoin(url, href)

                        if is_file_link(href):
                            anchor_text = (a.text or "").strip().lower()
                            tail = pathlib.PurePosixPath(urlparse(full).path).name.lower()
                            if anchor_text == filename_lower or tail == filename_lower:
                                logger.info(f"File located via crawl: {full}")
                                return full
                        if is_dir_link(href):
                            stack.append(full)

                logger.debug("Crawl finished — no file found.")
                return None

            for alias in aliases:
                start_url = urljoin(base, f"v/{alias}/browse")
                logger.debug(f"Crawling alias: {alias} -> {start_url}")
                found = crawl(start_url)
                if not found:
                    continue

                r = fetch(found, stream=True)
                if not r:
                    continue

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                try:
                    total = int(r.headers.get("content-length", 0)) or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Downloaded via crawl: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download during crawl failed: {e}")
                    return None

                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Youtube_Helper.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename, resolution, fps

            logger.warning(f"File '{filename_with_ext}' not found in any alias.")
            return None

    def download_video_with_resolution(self, vid, resolutions=["720p", "480p", "360p", "144p"], output_path="."):
        """
        Downloads a YouTube video in one of the specified resolutions and returns video details.

        This function attempts to download the video using the pytubefix/YouTube method.

        Parameters:
            vid (str): The YouTube video ID.
            resolutions (list of str, optional): A list of preferred video resolutions.
            output_path (str, optional): The directory where the video will be downloaded.

        Returns:
            tuple or None: A tuple (video_file_path, vid, resolution, fps) if successful,
                            or None if methods fail.
        """

        try:
            # Optionally upgrade pytubefix (if configured and it is Monday)
            if self.update_package and datetime.datetime.today().weekday() == 0:
                self.upgrade_package_if_needed("pytube")
                self.upgrade_package_if_needed("pytubefix")

            # Construct the YouTube URL.
            youtube_url = f"https://www.youtube.com/watch?v={vid}"

            # Create a YouTube object using the provided client configuration.
            if self.need_authentication:
                youtube_object = YouTube(youtube_url,
                                         self.client,
                                         use_oauth=True,
                                         allow_oauth_cache=True,
                                         on_progress_callback=on_progress)
            else:
                youtube_object = YouTube(youtube_url,
                                         self.client,
                                         on_progress_callback=on_progress)

            selected_stream = None
            selected_resolution = None

            # Iterate over the preferred resolutions to find a matching stream.
            for resolution in resolutions:
                # Filter the streams for the current resolution.
                video_streams = youtube_object.streams.filter(res=resolution)
                if video_streams:
                    selected_resolution = resolution
                    logger.debug(f"Found video {vid} in {resolution}.")
                    # Use the first available stream.
                    if hasattr(video_streams, 'first'):
                        selected_stream = video_streams.first()
                    else:
                        selected_stream = video_streams[0]
                    break

            # --- Minimal-change fallback block (added) ---
            if not selected_stream:
                # Fallback: auto-pick the highest available 'p' (progressive preferred) <= 720p

                def _height_from_res(res_str: str) -> int:
                    # Extract numeric height from strings like '720p', '1080p60', '576p'
                    if not res_str:
                        return -1
                    m = re.search(r'(\d{3,4})p', res_str)
                    return int(m.group(1)) if m else -1

                streams = youtube_object.streams
                progressive_candidates = []
                adaptive_candidates = []

                for s in streams:
                    # resolution attribute can be 'resolution' or 'res' depending on pytube variant
                    res_attr = getattr(s, "resolution", None) or getattr(s, "res", None)
                    h = _height_from_res(res_attr)  # type: ignore
                    if 0 < h <= 720:
                        mime = getattr(s, "mime_type", "") or ""
                        if "mp4" not in mime.lower():
                            continue
                        if getattr(s, "is_progressive", False):
                            progressive_candidates.append((h, s))
                        else:
                            adaptive_candidates.append((h, s))

                chosen = None
                if progressive_candidates:
                    chosen = max(progressive_candidates, key=lambda t: t[0])[1]
                elif adaptive_candidates:
                    chosen = max(adaptive_candidates, key=lambda t: t[0])[1]

                if chosen:
                    selected_stream = chosen
                    selected_resolution = getattr(chosen, "resolution", None) or getattr(chosen, "res", None)
                    logger.debug(f"{vid}: no preferred match; picked highest available {selected_resolution} (≤720p).")
                else:
                    logger.error(f"{vid}: no stream available ≤ 720p.")
                    return None
            # --- End fallback block ---

            # Construct the file path for the downloaded video.
            video_file_path = os.path.join(output_path, f"{vid}.mp4")
            logger.info(f"{vid}: download in {selected_resolution} started with pytubefix.")

            # Download the video.
            selected_stream.download(output_path, filename=f"{vid}.mp4")
            self.video_title = youtube_object.title
            fps = self.get_video_fps(video_file_path)
            logger.info(f"{vid}: FPS={fps}.")

            return video_file_path, vid, selected_resolution, fps

        except Exception as e:
            logger.error(f"{vid}: pytubefix download method failed: {e}")
            logger.info(f"{vid}: falling back to yt_dlp method.")

        # ----- Fallback method: using yt_dlp -----
        try:
            # Optionally upgrade yt_dlp (if the configuration requires it and it is Monday)
            if self.update_package and datetime.datetime.today().weekday() == 0:
                self.upgrade_package_if_needed("yt_dlp")

            # Construct the YouTube URL.
            youtube_url = f"https://www.youtube.com/watch?v={vid}"

            # Extract video information (including available formats) without downloading.
            extract_opts = {
                'skip_download': True,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(extract_opts) as ydl:  # pyright: ignore[reportArgumentType]
                info_dict = ydl.extract_info(youtube_url, download=False)

            available_formats: list[dict[str, Any]] = info_dict.get("formats") or []
            selected_format_str = None
            selected_resolution = None

            # Iterate over the preferred resolutions.
            for res in resolutions:
                try:
                    res_height = int(res.rstrip("p"))
                except ValueError:
                    continue

                # Check for a video-only stream (no audio).
                video_only_found = any(
                    fmt for fmt in available_formats
                    if fmt.get("height") == res_height and fmt.get("acodec") == "none"
                )
                if video_only_found:
                    selected_format_str = f"bestvideo[height={res_height}]"
                    selected_resolution = res
                    logger.info(f"{vid}: found video-only format in {res}.")
                    break

                # Otherwise, check for any stream at that resolution.
                progressive_found = any(
                    fmt for fmt in available_formats
                    if fmt.get("height") == res_height
                )
                if progressive_found:
                    selected_format_str = f"best[height={res_height}]"
                    selected_resolution = res
                    logger.info(f"{vid}: found progressive format in {res}. Audio will be removed.")
                    break

            if not selected_format_str:
                logger.error(f"{vid}: no stream available in the specified resolutions.")
                # Raise an exception to trigger the fallback method.
                raise Exception(f"{vid}: no stream available via yt_dlp")
            po_token = common.get_secrets("po_token")

            # Set download options.
            download_opts = {
                'format': selected_format_str,
                'outtmpl': os.path.join(output_path, f"{vid}.%(ext)s"),
                'quiet': True,
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4'
                }],
                'postprocessor_args': ['-an'],
                'http_headers': {
                    'Cookie': f'pot={po_token}'
                },
                'extractor_args': {
                    'youtube': {
                        'po_token': f'web.gvs+{po_token}'
                    }
                },
                'cookiesfrombrowser': ('chrome',)
            }

            logger.info(f"{vid}: download in {selected_resolution} started with yt_dlp.")
            with yt_dlp.YoutubeDL(download_opts) as ydl:  # type: ignore
                ydl.download([youtube_url])

            # Final output file path (assuming the postprocessor outputs an MP4 file).
            video_file_path = os.path.join(output_path, f"{vid}.mp4")
            self.video_title = info_dict.get("title")  # type: ignore
            fps = self.get_video_fps(video_file_path)
            logger.info(f"FPS of {vid}: {fps}.")

            return video_file_path, vid, selected_resolution, fps

        except Exception as e:
            logger.error(f"{vid}: yt_dlp download method failed: {e}.")
            return None

    def get_video_fps(self, video_file_path):
        """
        Retrieves the frames per second (FPS) of a video file using OpenCV.

        Parameters:
            video_file_path (str): The file path to the video whose FPS is to be determined.

        Returns:
            int or None: The rounded FPS value of the video if successful; otherwise, returns None if an error occurs.

        The function performs the following steps:
            1. Opens the video file using OpenCV's VideoCapture.
            2. Retrieves the FPS using the CAP_PROP_FPS property.
            3. Rounds the FPS value to the nearest integer.
            4. Releases the video resource.
            5. Returns the rounded FPS value, or None if an exception is encountered.
        """
        try:
            # Open the video file using OpenCV
            video = cv2.VideoCapture(video_file_path)

            # Retrieve the FPS using OpenCV's CAP_PROP_FPS property
            fps = video.get(cv2.CAP_PROP_FPS)

            # Release the video resource
            video.release()

            # Return the FPS rounded to the nearest integer
            return round(fps, 0)
        except Exception as e:
            # Log an error message if FPS retrieval fails
            logger.error(f"Failed to retrieve FPS: {e}")
            return None

    @staticmethod
    def get_video_resolution_label(video_path: str) -> str:
        """Return the resolution label (e.g., '720p', '1080p') for a given video file.

        This method inspects the video file to determine its frame height and then
        maps it to a common resolution label. If the resolution does not match a
        well-known standard, it falls back to returning `<height>p`.

        Args:
            video_path (str): Path to the video file.

        Returns:
            str: Resolution label (e.g., "720p", "1080p", "2160p").
                 Falls back to "<height>p" if no predefined label exists.

        Raises:
            FileNotFoundError: If the provided video path does not exist.
            RuntimeError: If the video file cannot be opened with OpenCV.
        """
        # Ensure the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Extract video frame height (resolution height in pixels)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Map common video resolutions to human-readable labels
        labels = {
            144: "144p",
            240: "240p",
            360: "360p",
            480: "480p",
            720: "720p",
            1080: "1080p",
            1440: "1440p",
            2160: "2160p",  # 4K UHD
        }

        # Return label if known, otherwise fallback to "<height>p"
        return labels.get(height, f"{height}p")

    def trim_video(self, input_path, output_path, start_time, end_time):
        """
        Trims a segment from a video and saves the result to a specified file.

        Parameters:
            input_path (str): The file path to the original video.
            output_path (str): The destination file path where the trimmed video will be saved.
            start_time (float or str): The start time for the trimmed segment. This can be specified in seconds
                                       or in a time format recognised by MoviePy.
            end_time (float or str): The end time for the trimmed segment. Similar to start_time, it can be in seconds
                                     or another supported time format.

        Returns:
            None

        The function performs the following steps:
            1. Loads the original video using MoviePy's VideoFileClip.
            2. Creates a subclip from the original video based on the provided start_time and end_time.
            3. Writes the subclip to the output_path using the H.264 video codec and AAC audio codec.
            4. Closes the video file to free up resources.
        """
        # Load the video and create a subclip using the provided start and end times.
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the subclip to the specified output file using the 'libx264' codec for video and 'aac' for audio.
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close the video clip to release any resources used.
        video_clip.close()

    def draw_yolo_boxes_on_video(self, df, fps, video_path, output_path):
        """
        Draw YOLO-style bounding boxes on a video and save the annotated output.

        This method takes a DataFrame containing normalised bounding box coordinates (in YOLO format),
        matches them frame-by-frame to the input video, draws the corresponding boxes and labels,
        and writes the resulting video to disk.

        Args:
            df (pd.DataFrame): DataFrame containing at least the following columns:
                - 'frame-count': Original frame indices in the source video.
                - 'X-center', 'Y-center': Normalised center coordinates (0 to 1).
                - 'width', 'Height': Normalised width and height (0 to 1).
                - 'unique-id': Identifier to display in the label.
            fps (float): Frames per second for the output video.
            video_path (str): Path to the input video file.
            output_path (str): Path to save the annotated output video.

        Raises:
            IOError: If the input video cannot be opened.
        """

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Normalise frame indices to start from 0
        min_frame = df["frame-count"].min()
        df["Frame Index"] = df["frame-count"] - min_frame

        # Attempt to open the input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Get video dimensions and total number of frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set up video writer with the same resolution and specified fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Writing to {output_path} ({width}x{height} @ {fps}fps)")

        frame_index = 0

        # Process each frame
        while frame_index < total_frames:
            success, frame = cap.read()
            if not success:
                logger.error(f"Failed to read frame {frame_index}")
                break

            # Filter YOLO data for this adjusted frame index
            frame_data = df[df["Frame Index"] == frame_index]

            for _, row in frame_data.iterrows():
                # Convert normalised coordinates to absolute pixel values
                x_center = row["x-center"] * width
                y_center = row["y-center"] * height
                w = row["width"] * width
                h = row["height"] * height

                # Calculate top-left and bottom-right corners of the box
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Draw rectangle and label with unique ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID: {int(row['unique-id'])}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the modified frame to the output video
            out.write(frame)
            frame_index += 1

        # Release video objects
        cap.release()
        out.release()

    def detect_gpu(self):
        """
        Detects whether an NVIDIA or Intel GPU is available and returns the appropriate FFmpeg encoder.

        Returns:
            str: 'hevc_nvenc' for NVIDIA, 'hevc_qsv' for Intel, or None if no compatible GPU is found.
        """
        try:
            # Check for NVIDIA GPU
            nvidia_check = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if nvidia_check.returncode == 0:
                return "hevc_nvenc"

            # Check for Intel QuickSync GPU
            intel_check = subprocess.run(["vainfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "Intel" in intel_check.stdout:
                return "hevc_qsv"

        except FileNotFoundError:
            pass  # Command not found, meaning the hardware isn't available or the driver isn't installed

        return None  # No compatible GPU found

    def compress_video(self, input_path, codec="libx265", preset="slow", crf=17):
        """
        Compresses a video using codec=codec.

        Args:
            input_path (str): Path to the input video file.
            codec (str, optional): Codec to use. Use H.265 by default.
            preset (str, optional): Value for preset.
            crf (int, optional): Value for crf. 17 is supposed to keep good quality with high level of compression.

        Returns:
            str: Path to the compressed video.

        Raises:
            e: error.
            FileNotFoundError: If the input video file does not exist.
            RuntimeError: If the compression process fails.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Extract filename and create output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(common.root_dir, filename)

        # Detect available GPU and set appropriate encoder
        codec_hw = self.detect_gpu()
        if codec_hw:
            codec = codec_hw  # Use detected hardware

        # Construct ffmpeg command
        command = [
            "ffmpeg",
            "-i", input_path,       # Input file
            "-c:v", codec,          # Use appropriate codec
            "-preset", preset,      # Compression speed/efficiency tradeoff
            "-crf", str(crf),       # Constant Rate Factor (lower = better quality, larger file)
            output_path,            # Temporary output file
            "-progress", "pipe:1",  # Enables real-time progress output
            "-nostats"              # Suppresses extra logs
        ]

        try:
            # Run ffmpeg command
            video_id = self.extract_youtube_id(input_path)
            logger.info(f"Started compression of {video_id} with {codec} codec. Current size={os.path.getsize(input_path)}.")  # noqa: E501
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            logger.info(f"Finished compression of {video_id} with {codec} codec. New size={os.path.getsize(output_path)}.")  # noqa: E501
            # Replace the original file with the compressed file
            shutil.move(output_path, input_path)
        except Exception as e:
            # Clean up temporary file in case of unexpected errors
            if os.path.exists(output_path):
                os.remove(output_path)
            logger.error(f"Video compression failed: {e.stderr.decode()}. Using uncompressed file.")  # type: ignore

    def extract_youtube_id(self, file_path):
        """
        Extracts the YouTube video ID from a given file path.

        Args:
            file_path (str): The full path of the video file.

        Returns:
            str: The extracted YouTube ID.

        Raises:
            ValueError: If no valid YouTube ID is found.
        """
        filename = os.path.basename(file_path)  # Get only the filename
        youtube_id, ext = os.path.splitext(filename)  # Remove the file extension

        if not youtube_id or len(youtube_id) < 5:  # Basic validation
            raise ValueError("Invalid YouTube ID extracted.")

        return youtube_id

    def create_video_from_images(self, image_folder, output_path, video_title, seg_mode=False, bbox_mode=False,
                                 frame_rate=30):
        """
        Creates a video file from a sequence of image frames.
        The output filename will reflect the mode used.

        Parameters:
            image_folder (str): Folder containing frame images.
            output_path (str): Directory where the output video will be saved.
            video_title (str): Base title for the video.
            seg_mode (bool): Whether segmentation mode is used.
            bbox_mode (bool): Whether bounding box mode is used.
            frame_rate (int or float): Frame rate for the video.
        """
        # Decide on the output filename based on mode
        if bbox_mode:
            output_filename = f"{video_title}_mod_bbox.mp4"
        elif seg_mode:
            output_filename = f"{video_title}_mod_seg.mp4"
        else:
            output_filename = f"{video_title}_mod.mp4"

        output_video_path = os.path.join(output_path, output_filename)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        def extract_frame_number(filename):
            match = re.search(r'frame_tracked_(\d+)\.jpg', filename)
            return int(match.group(1)) if match else float('inf')  # Push invalid ones to the end

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

        if not images:
            logger.error("No JPG images found in the specified folder.")
            return

        images.sort(key=extract_frame_number)

        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)

        if frame is None:
            logger.error(f"Could not read the first image: {first_image_path}")
            return

        height, width, _ = frame.shape

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
        return output_video_path

    def merge_txt_to_csv_dynamically_bbox(self, txt_location, output_csv, frame_count):
        """
        Merges YOLO-format label data from a .txt file into a CSV, frame by frame.

        Parameters:
            txt_location (str): Directory containing label .txt files.
            output_csv (str): Path to the CSV file to update.
            frame_count (int): Frame number being processed (used for naming).
        """
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
                         names=["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence"])
        df['frame-count'] = frame_count

        # Append the DataFrame to the CSV file
        if not os.path.exists(output_csv):
            df.to_csv(output_csv, index=False, mode='w')  # If the CSV does not exist, create it
        else:
            df.to_csv(output_csv, index=False, mode='a', header=False)  # If it exists, append without header

    def merge_txt_to_csv_dynamically_seg(self, txt_location, output_csv, frame_count):
        """
        Merges YOLO-format segmentation+tracking label data from a .txt file into a CSV, frame by frame.
        Handles possible formatting issues gracefully.
        """

        new_txt_file_name = os.path.join(txt_location, f"label_{frame_count}.txt")
        if not os.path.isfile(new_txt_file_name) or os.stat(new_txt_file_name).st_size == 0:
            return  # No labels for this frame

        rows = []
        with open(new_txt_file_name, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # skip malformed line
                try:
                    class_id = int(parts[0])
                    # Everything except the first (class_id) and last (track_id) are mask coordinates
                    mask_coords = parts[1:-2]
                    track_id = int(parts[-2])
                    confidence = float(parts[-1])
                    mask_points = " ".join(mask_coords)
                    rows.append([class_id, mask_points, track_id, confidence, frame_count])
                except Exception:
                    continue

        if not rows:
            return

        df = pd.DataFrame(rows, columns=["yolo-id", "mask-polygon", "unique-id", "confidence", "frame-count"])

        # Append the DataFrame to the CSV file
        if not os.path.exists(output_csv):
            df.to_csv(output_csv, index=False, mode='w')
        else:
            df.to_csv(output_csv, index=False, mode='a', header=False)

    def delete_folder(self, folder_path):
        """
        Deletes the folder and all its contents recursively.

        Parameters:
            folder_path (str): The path of the folder to delete.

        Returns:
            bool: True if the folder was successfully deleted, False otherwise.
        """
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                logger.info(f"Folder '{folder_path}' deleted successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to delete folder '{folder_path}': {e}")
                return False
        else:
            logger.info(f"Folder '{folder_path}' does not exist.")
            return False

    def delete_youtube_mod_videos(self, folders):
        """
        Deletes files of the form {youtube_id}_mod.mp4
        from the given list of folders.

        Args:
            folders (list): List of folder paths to scan.
        """
        pattern = re.compile(r"^[A-Za-z0-9_-]{11}_mod\.mp4$")

        for folder in folders:
            if not os.path.exists(folder):
                logger.info(f"Skipping missing folder: {folder}")
                continue

            for filename in os.listdir(folder):
                if pattern.match(filename):
                    file_path = os.path.join(folder, filename)
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted: {file_path}")
                    except Exception as e:
                        logger.info(f"Failed to delete {file_path}: {e}")

    def get_iso_alpha_3(self, country_name, existing_iso):
        """
        Converts a country name to ISO 3166-1 alpha-3 format.

        Parameters:
            country_name (str): Full country name.
            existing_iso (str): Existing ISO code as fallback.

        Returns:
            str or None: ISO 3166-1 alpha-3 code or fallback value.
        """
        try:
            return pycountry.countries.lookup(country_name).alpha_3
        except LookupError:
            if country_name.strip().upper() == "KOSOVO":
                return "XKX"  # User-assigned code for Kosovo
            return existing_iso if existing_iso else None

    def get_latest_population(self):
        """
        Fetches the latest available population data from World Bank.

        Returns:
            pd.DataFrame: Population data with columns ['iso3', 'Year', 'Population'].
        """
        # Search for the population indicator
        indicator = 'SP.POP.TOTL'  # Total Population (World Bank indicator code)

        # Fetch the latest population data
        population_data = wb.get_series(indicator, id_or_value='id', mrv=1)

        # Convert the data to a DataFrame
        population_df = population_data.reset_index()

        # Rename columns appropriately
        population_df = population_df.rename(columns={
            population_df.columns[0]: 'iso3',
            population_df.columns[2]: 'Year',
            population_df.columns[3]: 'Population'
        })

        # Divide population by 1000
        population_df['Population'] = population_df['Population'] / 1000

        return population_df

    def update_population_in_csv(self, data):
        """
        Updates the mapping DataFrame with the latest country population data.

        Parameters:
            data (pd.DataFrame): The mapping DataFrame to update.
        """

        # Ensure the required columns exist in the CSV
        if "iso3" not in data.columns:
            raise KeyError("The CSV file does not have a 'iso3' column.")

        if "population_country" not in data.columns:
            data["population_country"] = None  # Initialise the column if it doesn't exist

        # Get the latest population data
        latest_population = self.get_latest_population()

        # Create a dictionary for quick lookup
        population_dict = dict(zip(latest_population['iso3'], latest_population['Population']))

        # Update the population_country column
        for index, row in data.iterrows():
            iso3 = row["iso3"]
            population = population_dict.get(iso3, None)
            # Always update with the latest population data
            data.at[index, "population_country"] = population

        # Save the updated DataFrame back to the same CSV
        data.to_csv(self.mapping, index=False)
        logger.info("Mapping file updated successfully with country population.")

    def get_continent_from_country(self, country):
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

    def get_upload_date(self, video_id):
        """
        Retrieves the upload date of a YouTube video given its video ID.

        Parameters:
            video_id (str): YouTube video ID.

        Returns:
            str or None: Upload date in 'ddmmyyyy' format or None if not retrievable.
        """
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

    def get_latest_traffic_mortality(self):
        """
        Fetch the latest traffic mortality data from the World Bank.

        Returns:
            pd.DataFrame: DataFrame with iso3 and the latest Traffic Mortality Rate.
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

    def fill_traffic_mortality(self, df):
        """
        Fill the traffic mortality rate column in a CSV file using World Bank data.

        Args:
            file_path (str): Path to the input CSV file.
        """
        try:
            # Ensure the required columns exist
            if 'iso3' not in df.columns or 'traffic_mortality' not in df.columns:
                logger.error("The required columns 'iso3' and 'traffic_mortality' are missing from the file.")
                return

            # Get the latest traffic mortality data
            traffic_df = self.get_latest_traffic_mortality()

            # Merge the traffic mortality data with the existing DataFrame
            updated_df = pd.merge(df, traffic_df, on='iso3', how='left', suffixes=('', '_new'))

            # Update the traffic_mortality column with the new data
            updated_df['traffic_mortality'] = updated_df['traffic_mortality_new'].combine_first(
                updated_df['traffic_mortality'])

            # Drop the temporary column
            updated_df = updated_df.drop(columns=['traffic_mortality_new'])

            # Update in-memory mapping
            self.mapping = updated_df

            # Persist to the original CSV path
            mapping_path = common.get_configs("mapping")  # <- get the path again
            updated_df.to_csv(mapping_path, index=False)

            logger.info("Mapping file updated successfully with traffic mortality rate.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def get_latest_gini_values(self):
        """
        Fetch the latest GINI index data from the World Bank.

        Returns:
            pd.DataFrame: DataFrame with iso3 and the latest GINI index values.
        """
        # World Bank indicator for GINI index
        indicator = 'SI.POV.GINI'  # GINI index

        # Fetch the most recent data (mrv=1 for the most recent value)
        geni_data = wb.get_series(indicator, id_or_value='id', mrv=1)

        # Convert the data to a DataFrame
        geni_df = geni_data.reset_index()

        # Rename columns appropriately
        geni_df = geni_df.rename(columns={
            geni_df.columns[0]: 'iso3',
            geni_df.columns[2]: 'Year',
            geni_df.columns[3]: 'gini'
        })

        # Keep only the latest value for each country
        geni_df = geni_df.sort_values(by=['iso3', 'Year'],
                                      ascending=[True, False]).drop_duplicates(subset=['iso3'])

        return geni_df[['iso3', 'gini']]

    def fill_gini_data(self, df):
        """
        Fill the GINI index column in a CSV file using World Bank data.

        Args:
            file_path (str): Path to the input CSV file.
        """
        try:
            # Ensure the required column exists
            if 'gini' not in df.columns:
                logger.error("The required columns 'iso3' and/or 'gini' are missing from the file.")
                return

            # Get the latest GINI index data
            geni_df = self.get_latest_gini_values()

            # Merge the GINI index data with the existing DataFrame
            updated_df = pd.merge(df, geni_df, on='iso3', how='left', suffixes=('', '_new'))

            # Update the geni column with the new data
            updated_df['gini'] = updated_df['gini_new'].combine_first(updated_df['gini'])

            # Drop the temporary column
            updated_df = updated_df.drop(columns=['gini_new'])

            # Update in-memory mapping
            self.mapping = updated_df

            # Persist to the original CSV path
            mapping_path = common.get_configs("mapping")  # path to the mapping CSV
            updated_df.to_csv(mapping_path, index=False)
            logger.info("Mapping file updated successfully with GINI value.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def update_track_buffer_in_yaml(self, yaml_path, video_fps):
        # Load existing YAML
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update the track_buffer value
        config['track_buffer'] = common.get_configs("track_buffer_sec") * video_fps

        # Write it back to the YAML file (overwrite)
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def tracking_mode(self, input_video_path, output_video_path, video_title, video_fps, seg_mode, bbox_mode, flag=0):
        """
        Performs object tracking on a video using YOLO models and saves results.
        (docstring as in your code)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.bbox_tracker == "bbox_custom_tracker.yaml" and bbox_mode is True:
            yaml_path = 'bbox_custom_tracker.yaml'
            self.update_track_buffer_in_yaml(yaml_path, video_fps)

        if self.seg_tracker == "seg_custom_tracker.yaml" and seg_mode is True:
            yaml_path = 'seg_custom_tracker.yaml'
            self.update_track_buffer_in_yaml(yaml_path, video_fps)

        # --- Model Initialisation ---
        if bbox_mode:
            bbox_model = YOLO(self.tracking_model)
        if seg_mode:
            seg_model = YOLO(self.segment_model)

        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        # Output directories
        if bbox_mode:
            bbox_frames_output_path = os.path.join("runs", "detect", "frames")
            bbox_annotated_frame_output_path = os.path.join("runs", "detect", "annotated_frames")
            bbox_tracked_frame_output_path = os.path.join("runs", "detect", "tracked_frame")
            bbox_txt_output_path = os.path.join("runs", "detect", "labels")
            bbox_text_filename = os.path.join("runs", "detect", "track", "labels", "image0.txt")
            bbox_display_video_output_path = os.path.join("runs", "detect", "display_video.mp4")
            os.makedirs(bbox_frames_output_path, exist_ok=True)
            os.makedirs(bbox_txt_output_path, exist_ok=True)
            os.makedirs(bbox_annotated_frame_output_path, exist_ok=True)
            os.makedirs(bbox_tracked_frame_output_path, exist_ok=True)

        if seg_mode:
            seg_frames_output_path = os.path.join("runs", "segment", "frames")
            seg_annotated_frame_output_path = os.path.join("runs", "segment", "annotated_frames")
            seg_tracked_frame_output_path = os.path.join("runs", "segment", "tracked_frame")
            seg_txt_output_path = os.path.join("runs", "segment", "labels")
            seg_text_filename = os.path.join("runs", "segment", "track", "labels", "image0.txt")
            seg_display_video_output_path = os.path.join("runs", "segment", "display_video.mp4")
            os.makedirs(seg_frames_output_path, exist_ok=True)
            os.makedirs(seg_txt_output_path, exist_ok=True)
            os.makedirs(seg_annotated_frame_output_path, exist_ok=True)
            os.makedirs(seg_tracked_frame_output_path, exist_ok=True)

        # Video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore

        if bbox_mode and self.display_frame_tracking:
            bbox_video_writer = cv2.VideoWriter(
                bbox_display_video_output_path, fourcc, video_fps, (frame_width, frame_height)
            )
        if seg_mode and self.display_frame_segmentation:
            seg_video_writer = cv2.VideoWriter(
                seg_display_video_output_path, fourcc, video_fps, (frame_width, frame_height)
            )

        # Progress bar
        if total_frames == 0:
            logger.warning("Warning: Could not determine total frames. Progress bar may not work correctly.")
            total_frames = None

        progress_bar = tqdm(total=total_frames, unit="frames", dynamic_ncols=True)
        frame_count = 0

        # Track history
        if seg_mode:
            seg_track_history = defaultdict(lambda: [])
        if bbox_mode:
            bbox_track_history = defaultdict(lambda: [])

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # -------- SEGMENTATION MODE --------
            seg_failed = False
            seg_boxes_xywh = None
            seg_track_ids = []
            seg_confidences = []
            seg_annotated_frame = frame.copy()
            # Track attempt
            if seg_mode:
                try:
                    seg_results = seg_model.track(
                        frame,
                        tracker=self.seg_tracker,
                        persist=True,
                        conf=self.confidence,
                        save=True,
                        save_txt=True,
                        line_width=LINE_THICKNESS,
                        show_labels=SHOW_LABELS,
                        show_conf=SHOW_CONF,
                        show=RENDER,
                        verbose=False,
                        device=device,
                    )
                    seg_boxes_obj = seg_results[0].boxes
                    seg_boxes_xywh = seg_boxes_obj.xywh.cpu() if seg_boxes_obj is not None else None  # type: ignore
                    if seg_boxes_xywh is not None and seg_boxes_xywh.size(0) > 0:
                        seg_track_ids = (
                            seg_boxes_obj.id.int().cpu().tolist()  # type: ignore
                            if hasattr(seg_boxes_obj, "id") and seg_boxes_obj.id is not None  # type: ignore
                            else []
                        )
                        seg_confidences = (
                            seg_boxes_obj.conf.cpu().tolist()  # type: ignore
                            if hasattr(seg_boxes_obj, "conf") and seg_boxes_obj.conf is not None  # type: ignore
                            else []
                        )
                        seg_annotated_frame = seg_results[0].plot()
                    else:
                        logger.info(f"[Frame {frame_count}] Segmentation: No objects found. Using original frame.")
                        with open(seg_text_filename, 'w') as file:
                            pass
                except Exception as e:
                    logger.error(f"[Frame {frame_count}] Segmentation failed: {e}. Using original frame.")
                    seg_failed = True

            # -------- BBOX MODE --------
            bbox_failed = False
            bbox_boxes_xywh = None
            bbox_track_ids = []
            bbox_confidences = []
            bbox_annotated_frame = frame.copy()
            if bbox_mode:
                try:
                    bbox_results = bbox_model.track(
                        frame,
                        tracker=self.bbox_tracker,
                        persist=True,
                        conf=self.confidence,
                        save=True,
                        save_txt=True,
                        line_width=LINE_THICKNESS,
                        show_labels=SHOW_LABELS,
                        show_conf=SHOW_CONF,
                        show=RENDER,
                        verbose=False,
                        device=device,
                    )
                    bbox_boxes_obj = bbox_results[0].boxes
                    bbox_boxes_xywh = bbox_boxes_obj.xywh.cpu() if bbox_boxes_obj is not None else None  # type: ignore
                    if bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                        bbox_track_ids = (
                            bbox_boxes_obj.id.int().cpu().tolist()  # type: ignore
                            if hasattr(bbox_boxes_obj, "id") and bbox_boxes_obj.id is not None  # type: ignore
                            else []
                        )
                        bbox_confidences = (
                            bbox_boxes_obj.conf.cpu().tolist()  # type: ignore
                            if hasattr(bbox_boxes_obj, "conf") and bbox_boxes_obj.conf is not None  # type: ignore
                            else []
                        )
                        bbox_annotated_frame = bbox_results[0].plot()
                    else:
                        logger.info(f"[Frame {frame_count}] BBox: No objects found. Using original frame.")
                        with open(bbox_text_filename, 'w') as file:  # noqa:F841
                            pass
                except Exception as e:
                    logger.error(f"[Frame {frame_count}] BBox failed: {e}. Using original frame.")
                    bbox_failed = True

            progress_bar.update(1)

            # Save annotated frames
            if self.save_annoted_img:
                if seg_mode:
                    seg_frame_filename = os.path.join(seg_annotated_frame_output_path, f"frame_{frame_count}.jpg")
                    cv2.imwrite(seg_frame_filename, seg_annotated_frame)
                if bbox_mode:
                    bbox_frame_filename = os.path.join(bbox_annotated_frame_output_path, f"frame_{frame_count}.jpg")
                    cv2.imwrite(bbox_frame_filename, bbox_annotated_frame)

            # Save txt files (only if detections found and not failed)
            if seg_mode and not seg_failed and seg_boxes_xywh is not None and seg_boxes_xywh.size(0) > 0:
                with open(seg_text_filename, 'r') as seg_text_file:
                    seg_data = seg_text_file.readlines()
                new_txt_file_name_seg = os.path.join("runs", "segment", "labels", f"label_{frame_count}.txt")

                if len(seg_data) != len(seg_confidences):
                    logger.warning(f"Warning: Number of bbox lines ({len(seg_data)}) does not match number of confidences ({len(seg_confidences)}).")  # noqa:E501

                with open(new_txt_file_name_seg, 'w') as seg_new_file:
                    for line, conf in zip(seg_data, seg_confidences):
                        line = line.rstrip('\n')
                        seg_new_file.write(f"{line} {conf:.6f}\n")

                seg_labels_path = os.path.join("runs", "segment", "labels")
                seg_output_csv_path = os.path.join("runs", "segment", f"{self.video_title}.csv")
                self.merge_txt_to_csv_dynamically_seg(seg_labels_path, seg_output_csv_path, frame_count)
                os.remove(seg_text_filename)

            if bbox_mode and not bbox_failed and bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                with open(bbox_text_filename, 'r') as bbox_text_file:
                    bbox_data = bbox_text_file.readlines()
                new_txt_file_name_bbox = os.path.join("runs", "detect", "labels", f"label_{frame_count}.txt")

                if len(bbox_data) != len(bbox_confidences):
                    logger.warning(f"Warning: Number of bbox lines ({len(bbox_data)}) does not match number of confidences ({len(bbox_confidences)}).")  # noqa:E501

                with open(new_txt_file_name_bbox, 'w') as bbox_new_file:
                    for line, conf in zip(bbox_data, bbox_confidences):
                        line = line.rstrip('\n')
                        bbox_new_file.write(f"{line} {conf:.6f}\n")
                bbox_labels_path = os.path.join("runs", "detect", "labels")
                bbox_output_csv_path = os.path.join("runs", "detect", f"{self.video_title}.csv")
                self.merge_txt_to_csv_dynamically_bbox(bbox_labels_path, bbox_output_csv_path, frame_count)
                os.remove(bbox_text_filename)

            if self.delete_labels is True:
                if seg_mode and not seg_failed and seg_boxes_xywh is not None and seg_boxes_xywh.size(0) > 0:
                    os.remove(os.path.join("runs", "segment", "labels", f"label_{frame_count}.txt"))
                if bbox_mode and not bbox_failed and bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                    os.remove(os.path.join("runs", "detect", "labels", f"label_{frame_count}.txt"))

            # Save labeled image
            if self.delete_frames is False:
                if seg_mode and not seg_failed and seg_boxes_xywh is not None and seg_boxes_xywh.size(0) > 0:
                    seg_image_filename = os.path.join("runs", "segment", "track", "image0.jpg")
                    seg_new_img_file_name = os.path.join("runs", "segment", "frames", f"frame_{frame_count}.jpg")
                    shutil.move(seg_image_filename, seg_new_img_file_name)
                if bbox_mode and not bbox_failed and bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                    bbox_image_filename = os.path.join("runs", "detect", "track", "image0.jpg")
                    bbox_new_img_file_name = os.path.join("runs", "detect", "frames", f"frame_{frame_count}.jpg")
                    shutil.move(bbox_image_filename, bbox_new_img_file_name)

            # Plot the tracks
            try:
                if seg_mode and not seg_failed and seg_boxes_xywh is not None and seg_boxes_xywh.size(0) > 0:
                    for box, track_id in zip(seg_boxes_xywh, seg_track_ids):
                        x, y, w, h = box
                        track = seg_track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(seg_annotated_frame,
                                      [points],
                                      isClosed=False,
                                      color=(230, 230, 230),
                                      thickness=LINE_THICKNESS * 5)

                if bbox_mode and not bbox_failed and bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                    for box, track_id in zip(bbox_boxes_xywh, bbox_track_ids):
                        x, y, w, h = box
                        track = bbox_track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(bbox_annotated_frame,
                                      [points],
                                      isClosed=False,
                                      color=(230, 230, 230),
                                      thickness=LINE_THICKNESS * 5)

            except Exception:
                pass

            # Display the annotated frame
            if self.display_frame_tracking:
                if seg_mode:
                    cv2.imshow("YOLOv11 Segmentation & Tracking", seg_annotated_frame)
                    seg_video_writer.write(seg_annotated_frame)
                if bbox_mode:
                    cv2.imshow("YOLOv11 Tracking", bbox_annotated_frame)
                    bbox_video_writer.write(bbox_annotated_frame)

            # Save the tracked frame here
            if self.save_tracked_img:
                if seg_mode:
                    seg_frame_filename = os.path.join(seg_tracked_frame_output_path,
                                                      f"frame_tracked_{frame_count}.jpg")
                    cv2.imwrite(seg_frame_filename, seg_annotated_frame)
                if bbox_mode:
                    bbox_frame_filename = os.path.join(bbox_tracked_frame_output_path,
                                                       f"frame_tracked_{frame_count}.jpg")
                    cv2.imwrite(bbox_frame_filename, bbox_annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        progress_bar.close()

        if flag:
            if seg_mode:
                self.create_video_from_images(
                    image_folder=seg_tracked_frame_output_path,
                    output_path=output_video_path,
                    video_title=video_title,
                    seg_mode=seg_mode,
                    frame_rate=video_fps,
                )
            if bbox_mode:
                self.create_video_from_images(
                    image_folder=bbox_tracked_frame_output_path,
                    output_path=output_video_path,
                    video_title=video_title,
                    bbox_mode=bbox_mode,
                    frame_rate=video_fps,
                )
