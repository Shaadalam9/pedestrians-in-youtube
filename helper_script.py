# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
# -----------------------------------------------------------------------------
# helper_script.py (thread-safe tracking + per-frame tqdm)
#
# Key additions vs legacy:
# - tracking_mode_threadsafe(): writes CSVs directly (same schema), no runs/ usage
# - Thread-local YOLO model cache: one model instance per worker thread
# - Thread-safe tracker YAML handling: per-job temp YAML copy for custom trackers
# - Optional per-segment ID remap so unique-id starts at 1 for each segment (default ON)
# - Optional per-frame tqdm progress bar (safe with multiple workers via position=)
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Optional, Set
from urllib.parse import urljoin, urlparse

import cv2
import pandas as pd
import requests
import torch
import world_bank_data as wb
import yt_dlp
from bs4 import BeautifulSoup
from moviepy.video.io.VideoFileClip import VideoFileClip
from pytubefix import YouTube
from pytubefix.cli import on_progress
from tqdm import tqdm
from ultralytics import YOLO

import common
from custom_logger import CustomLogger

logger = CustomLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Consts / defaults
LINE_THICKNESS = 1
RENDER = False
SHOW_LABELS = False
SHOW_CONF = False

UPGRADE_LOG_FILE = "upgrade_log.json"

# Thread-local storage for per-thread model instances
_TLS = threading.local()


class Youtube_Helper:
    """
    Helper class: download/trim/video utils + tracking.
    """

    def __init__(self, video_title: Optional[str] = None):
        self.video_title = video_title

        # Models / trackers (same keys you already use)
        self.tracking_model = common.get_configs("tracking_model")
        self.segment_model = common.get_configs("segment_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.seg_tracker = common.get_configs("seg_tracker")

        # Tracking behavior
        self.confidence = float(common.get_configs("confidence") or 0.0) if "confidence" in dir(common) else 0.0    # noqa: E501
        try:
            self.track_buffer_sec = float(common.get_configs("track_buffer_sec"))
        except Exception:
            self.track_buffer_sec = 2.0

        # Existing flags (kept for compatibility; not required by thread-safe path)
        self.display_frame_tracking = bool(common.get_configs("display_frame_tracking"))
        self.display_frame_segmentation = bool(common.get_configs("display_frame_segmentation"))
        self.output_path = common.get_configs("videos")
        self.save_annoted_img = bool(common.get_configs("save_annoted_img"))
        self.save_tracked_img = bool(common.get_configs("save_tracked_img"))
        self.delete_labels = bool(common.get_configs("delete_labels"))
        self.delete_frames = bool(common.get_configs("delete_frames"))
        self.update_package = bool(common.get_configs("update_package"))
        self.need_authentication = bool(common.get_configs("need_authentication"))
        self.client = common.get_configs("client")

        # Mapping (some pipelines still use this)
        try:
            self.mapping_path = common.get_configs("mapping")
            self.mapping = pd.read_csv(self.mapping_path)
        except Exception:
            self.mapping_path = None
            self.mapping = None

    # -------------------------------------------------------------------------
    # General utilities
    # -------------------------------------------------------------------------

    def set_video_title(self, title: str) -> None:
        self.video_title = title

    def rename_folder(self, old_name: str, new_name: str) -> None:
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            logger.error(f"Error: Folder '{old_name}' not found.")
        except FileExistsError:
            logger.error(f"Error: Folder '{new_name}' already exists.")

    # -------------------------------------------------------------------------
    # Package upgrade logging (kept)
    # -------------------------------------------------------------------------

    def load_upgrade_log(self) -> dict:
        if not os.path.exists(UPGRADE_LOG_FILE):
            return {}
        try:
            with open(UPGRADE_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_upgrade_log(self, log_data: dict) -> None:
        with open(UPGRADE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f)

    def was_upgraded_today(self, package_name: str) -> bool:
        log_data = self.load_upgrade_log()
        today = datetime.date.today().isoformat()
        return log_data.get(package_name) == today

    def mark_as_upgraded(self, package_name: str) -> None:
        log_data = self.load_upgrade_log()
        log_data[package_name] = datetime.date.today().isoformat()
        self.save_upgrade_log(log_data)

    def upgrade_package_if_needed(self, package_name: str) -> None:
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
            self.mark_as_upgraded(package_name)

    # -------------------------------------------------------------------------
    # File stability + safe copy (kept)
    # -------------------------------------------------------------------------

    @staticmethod
    def _wait_for_stable_file(src: str, checks: int = 2, interval: float = 0.5, timeout: float = 30) -> bool:
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

    def copy_video_safe(self, base_video_path: str, internal_ssd: str, vid: str,
                        max_attempts: int = 5, backoff: float = 0.6) -> str:
        if not vid or str(vid).strip() == "":
            raise ValueError("vid must be a non-empty string")

        dest_dir = internal_ssd
        dest = os.path.join(dest_dir, f"{vid}.mp4")
        tmp = dest + ".tmp"

        os.makedirs(dest_dir, exist_ok=True)

        try:
            if os.path.exists(dest) and os.path.exists(base_video_path) and os.path.samefile(base_video_path, dest):
                return dest
        except OSError:
            pass

        if not Youtube_Helper._wait_for_stable_file(base_video_path):
            raise FileNotFoundError(f"Source never became available or stayed unstable: {base_video_path}")

        try:
            src_size = os.stat(base_video_path).st_size
        except OSError as e:
            raise FileNotFoundError(f"Cannot stat source: {base_video_path}: {e!r}")

        attempt = 0
        while True:
            attempt += 1
            try:
                shutil.copy2(base_video_path, tmp)
                try:
                    with open(tmp, "rb") as f:
                        os.fsync(f.fileno())
                except Exception:
                    pass
                os.replace(tmp, dest)

                if os.stat(dest).st_size != src_size:
                    raise OSError(f"Size mismatch after copy: src={src_size}, dst={os.stat(dest).st_size}")

                return dest

            except (OSError, IOError):
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                if attempt >= max_attempts:
                    raise
                time.sleep(backoff * attempt)

    # -------------------------------------------------------------------------
    # Download: FTP-like HTTP file server (kept)
    # -------------------------------------------------------------------------

    def download_videos_from_ftp(
        self,
        filename: str,
        base_url: Optional[str] = None,
        out_dir: str = ".",
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 20,
        debug: bool = True,
        max_pages: int = 500,
    ) -> Optional[tuple[str, str, str, float]]:
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
        if debug:
            logger.debug(
                f"Base URL: {base} | Auth: {'Basic' if username and password else 'None'} | Token: {'Yes' if token else 'No'}"  # noqa: E501
            )

        with requests.Session() as session:
            if username and password:
                session.auth = (username, password)
            session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

            def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
                try:
                    r = session.get(url, timeout=timeout, params=req_params, stream=stream)
                    if debug:
                        logger.debug(f"GET {url} -> {r.status_code}")
                    if r.status_code == 401:
                        logger.error(f"Authentication failed for {url}")
                    r.raise_for_status()
                    return r
                except requests.RequestException as e:
                    logger.warning(f"Request failed [{url}]: {e}")
                    return None

            # 1) Try direct /files paths
            for alias in aliases:
                direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
                if debug:
                    logger.debug(f"Trying direct URL: {direct_url}")
                r = fetch(direct_url, stream=True)
                if r is None:
                    continue

                logger.info(f"Found file via direct URL: {direct_url}")
                content_len = int(r.headers.get("content-length", 0))

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                # avoid overwrite
                if os.path.exists(local_path):
                    stem, suf = os.path.splitext(local_path)
                    i = 1
                    while os.path.exists(f"{stem} ({i}){suf}"):
                        i += 1
                    local_path = f"{stem} ({i}){suf}"
                    logger.warning(f"File exists, saving as: {local_path}")

                try:
                    total = content_len or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading from ftp: {filename_with_ext}",
                        dynamic_ncols=True,
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

                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path) or 0.0)
                    resolution = self.get_video_resolution_label(local_path)
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename, resolution, fps

            # 2) Crawl /browse fallback
            visited: Set[str] = set()

            def is_dir_link(href: str) -> bool:
                return href.startswith("/v/") and "/browse" in href

            def is_file_link(href: str) -> bool:
                return "/files/" in href

            def crawl(start_url: str) -> Optional[str]:
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
                        href = (a.get("href") or "").strip()  # type: ignore
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

                return None

            for alias in aliases:
                start_url = urljoin(base, f"v/{alias}/browse")
                if debug:
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
                        dynamic_ncols=True,
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
                    fps = float(self.get_video_fps(local_path) or 0.0)
                    resolution = self.get_video_resolution_label(local_path)
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename, resolution, fps

            logger.warning(f"File '{filename_with_ext}' not found in any alias.")
            return None

    # -------------------------------------------------------------------------
    # Download: YouTube with resolution preference (kept)
    # -------------------------------------------------------------------------

    def download_video_with_resolution(
        self,
        vid: str,
        resolutions: list[str] = ["720p", "480p", "360p", "144p"],
        output_path: str = ".",
    ) -> Optional[tuple[str, str, str, float]]:
        try:
            if self.update_package and datetime.datetime.today().weekday() == 0:
                self.upgrade_package_if_needed("pytube")
                self.upgrade_package_if_needed("pytubefix")

            youtube_url = f"https://www.youtube.com/watch?v={vid}"
            if self.need_authentication:
                youtube_object = YouTube(
                    youtube_url,
                    self.client,
                    use_oauth=True,
                    allow_oauth_cache=True,
                    on_progress_callback=on_progress,
                )
            else:
                youtube_object = YouTube(youtube_url, self.client, on_progress_callback=on_progress)

            selected_stream = None
            selected_resolution = None

            for resolution in resolutions:
                streams = youtube_object.streams.filter(res=resolution)
                if streams:
                    selected_resolution = resolution
                    selected_stream = streams.first() if hasattr(streams, "first") else streams[0]
                    break

            # fallback: pick best <= 720p mp4
            if not selected_stream:

                def _height_from_res(res_str: str) -> int:
                    if not res_str:
                        return -1
                    m = re.search(r"(\d{3,4})p", res_str)
                    return int(m.group(1)) if m else -1

                progressive_candidates = []
                adaptive_candidates = []
                for s in youtube_object.streams:
                    res_attr = getattr(s, "resolution", None) or getattr(s, "res", None)
                    h = _height_from_res(res_attr or "")
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
                else:
                    logger.error(f"{vid}: no stream available ≤ 720p.")
                    return None

            video_file_path = os.path.join(output_path, f"{vid}.mp4")
            logger.info(f"{vid}: download in {selected_resolution} started with pytubefix.")
            selected_stream.download(output_path, filename=f"{vid}.mp4")

            self.video_title = youtube_object.title
            fps = self.get_video_fps(video_file_path)
            logger.info(f"{vid}: FPS={fps}.")
            return video_file_path, vid, str(selected_resolution), float(fps or 0.0)

        except Exception as e:
            logger.error(f"{vid}: pytubefix download failed: {e}")
            logger.info(f"{vid}: falling back to yt_dlp method.")

        # yt_dlp fallback
        try:
            if self.update_package and datetime.datetime.today().weekday() == 0:
                self.upgrade_package_if_needed("yt_dlp")

            youtube_url = f"https://www.youtube.com/watch?v={vid}"

            extract_opts = {"skip_download": True, "quiet": True}
            with yt_dlp.YoutubeDL(extract_opts) as ydl:  # type: ignore
                info_dict = ydl.extract_info(youtube_url, download=False)

            available_formats: list[dict[str, Any]] = info_dict.get("formats") or []
            selected_format_str = None
            selected_resolution = None

            for res in resolutions:
                try:
                    res_height = int(res.rstrip("p"))
                except ValueError:
                    continue

                video_only_found = any(
                    fmt for fmt in available_formats
                    if fmt.get("height") == res_height and fmt.get("acodec") == "none"
                )
                if video_only_found:
                    selected_format_str = f"bestvideo[height={res_height}]"
                    selected_resolution = res
                    break

                progressive_found = any(fmt for fmt in available_formats if fmt.get("height") == res_height)
                if progressive_found:
                    selected_format_str = f"best[height={res_height}]"
                    selected_resolution = res
                    break

            if not selected_format_str:
                logger.error(f"{vid}: no stream available in requested resolutions via yt_dlp.")
                raise RuntimeError("yt_dlp: no matching format")

            po_token = common.get_secrets("po_token")

            download_opts = {
                "format": selected_format_str,
                "outtmpl": os.path.join(output_path, f"{vid}.%(ext)s"),
                "quiet": True,
                "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
                "postprocessor_args": ["-an"],
                "http_headers": {"Cookie": f"pot={po_token}"},
                "extractor_args": {"youtube": {"po_token": f"web.gvs+{po_token}"}},
                "cookiesfrombrowser": ("chrome",),
            }

            logger.info(f"{vid}: download in {selected_resolution} started with yt_dlp.")
            with yt_dlp.YoutubeDL(download_opts) as ydl:  # type: ignore
                ydl.download([youtube_url])

            video_file_path = os.path.join(output_path, f"{vid}.mp4")
            self.video_title = info_dict.get("title")  # type: ignore
            fps = self.get_video_fps(video_file_path)
            logger.info(f"{vid}: FPS={fps}.")
            return video_file_path, vid, str(selected_resolution), float(fps or 0.0)

        except Exception as e:
            logger.error(f"{vid}: yt_dlp download failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Video info / trimming
    # -------------------------------------------------------------------------

    def get_video_fps(self, video_file_path: str) -> Optional[float]:
        try:
            video = cv2.VideoCapture(video_file_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            video.release()
            return float(round(fps, 0))
        except Exception as e:
            logger.error(f"Failed to retrieve FPS: {e}")
            return None

    def get_video_resolution_label(self, video_path: str) -> str:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        labels = {
            144: "144p", 240: "240p", 360: "360p", 480: "480p",
            720: "720p", 1080: "1080p", 1440: "1440p", 2160: "2160p"
        }
        if height in labels:
            return labels[height]

        tolerance = 8
        standard_heights = sorted(labels.keys())
        closest = min(standard_heights, key=lambda h: abs(h - height))
        if abs(closest - height) <= tolerance:
            return labels[closest]
        return f"{height}p"

    def trim_video(self, input_path: str, output_path: str, start_time: int, end_time: int) -> None:
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        video_clip.close()

    # -------------------------------------------------------------------------
    # Housekeeping
    # -------------------------------------------------------------------------

    def delete_folder(self, folder_path: str) -> bool:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                logger.info(f"Folder '{folder_path}' deleted successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to delete folder '{folder_path}': {e}")
                return False
        logger.info(f"Folder '{folder_path}' does not exist.")
        return False

    def delete_youtube_mod_videos(self, folders: list[str]) -> None:
        pattern = re.compile(r"^[A-Za-z0-9_-]{11}_.*_mod\.mp4$")
        for folder in folders:
            if not os.path.exists(folder):
                logger.info(f"Skipping missing folder: {folder}")
                continue
            for fn in os.listdir(folder):
                if pattern.match(fn):
                    fp = os.path.join(folder, fn)
                    try:
                        os.remove(fp)
                        logger.info(f"Deleted: {fp}")
                    except Exception as e:
                        logger.info(f"Failed to delete {fp}: {e}")

    # -------------------------------------------------------------------------
    # Mapping helpers (kept)
    # -------------------------------------------------------------------------

    def get_iso_alpha_3(self, country_name: str, existing_iso: Optional[str]) -> Optional[str]:
        try:
            import pycountry
            return pycountry.countries.lookup(country_name).alpha_3
        except Exception:
            if country_name.strip().upper() == "KOSOVO":
                return "XKX"
            return existing_iso if existing_iso else None

    def get_upload_date(self, video_id: str) -> Optional[str]:
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            yt = YouTube(video_url)
            upload_date = yt.publish_date
            return upload_date.strftime("%d%m%Y") if upload_date else None
        except Exception:
            return None

    def get_latest_population(self) -> pd.DataFrame:
        indicator = "SP.POP.TOTL"
        population_data = wb.get_series(indicator, id_or_value="id", mrv=1)
        df = population_data.reset_index()
        df = df.rename(columns={df.columns[0]: "iso3", df.columns[2]: "Year", df.columns[3]: "Population"})
        df["Population"] = df["Population"] / 1000
        return df

    def update_population_in_csv(self, data: pd.DataFrame) -> None:
        if "iso3" not in data.columns:
            raise KeyError("The CSV file does not have a 'iso3' column.")
        if "population_country" not in data.columns:
            data["population_country"] = None

        latest_population = self.get_latest_population()
        population_dict = dict(zip(latest_population["iso3"], latest_population["Population"]))
        for idx, row in data.iterrows():
            iso3 = row["iso3"]
            data.at[idx, "population_country"] = population_dict.get(iso3, None)  # type: ignore

        mapping_path = common.get_configs("mapping")
        data.to_csv(mapping_path, index=False)
        logger.info("Mapping file updated successfully with country population.")

    def get_latest_gini_values(self) -> pd.DataFrame:
        indicator = "SI.POV.GINI"
        gini_data = wb.get_series(indicator, id_or_value="id", mrv=1)
        df = gini_data.reset_index()
        df = df.rename(columns={df.columns[0]: "iso3", df.columns[2]: "Year", df.columns[3]: "gini"})
        df = df.sort_values(by=["iso3", "Year"], ascending=[True, False]).drop_duplicates(subset=["iso3"])
        return df[["iso3", "gini"]]

    def fill_gini_data(self, df: pd.DataFrame) -> None:
        try:
            if "iso3" not in df.columns:
                logger.error("Missing column 'iso3'.")
                return
            if "gini" not in df.columns:
                df["gini"] = None

            gini_df = self.get_latest_gini_values()
            updated_df = pd.merge(df, gini_df, on="iso3", how="left", suffixes=("", "_new"))
            updated_df["gini"] = updated_df["gini_new"].combine_first(updated_df["gini"])
            updated_df = updated_df.drop(columns=["gini_new"])

            mapping_path = common.get_configs("mapping")
            updated_df.to_csv(mapping_path, index=False)
            logger.info("Mapping file updated successfully with GINI value.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    # -------------------------------------------------------------------------
    # Thread-safe tracking helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _write_rows_csv(path: str, header: list[str], rows: list[list]) -> None:
        if not rows:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerows(rows)

    def get_thread_models(self, bbox_mode: bool, seg_mode: bool) -> tuple[Optional[YOLO], Optional[YOLO]]:
        """
        Thread-local cache: each worker thread gets its own YOLO model instances.
        This avoids race conditions inside Ultralytics trackers.
        """
        if bbox_mode:
            if not hasattr(_TLS, "bbox_model") or _TLS.bbox_model is None:
                _TLS.bbox_model = YOLO(self.tracking_model)
        else:
            _TLS.bbox_model = None

        if seg_mode:
            if not hasattr(_TLS, "seg_model") or _TLS.seg_model is None:
                _TLS.seg_model = YOLO(self.segment_model)
        else:
            _TLS.seg_model = None

        return _TLS.bbox_model, _TLS.seg_model

    def make_tracker_config(self, tracker_path: str, video_fps: int) -> str:
        """
        Thread-safe alternative to in-place YAML edits:
        - Only for bbox_custom_tracker.yaml / seg_custom_tracker.yaml
        - Creates a temp copy with track_buffer = track_buffer_sec * fps
        """
        if not (isinstance(tracker_path, str) and tracker_path.endswith(".yaml") and os.path.isfile(tracker_path)):
            return tracker_path

        base = os.path.basename(tracker_path)
        if base not in ("bbox_custom_tracker.yaml", "seg_custom_tracker.yaml"):
            return tracker_path

        try:
            import yaml
        except Exception:
            return tracker_path

        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return tracker_path

        cfg["track_buffer"] = float(self.track_buffer_sec) * float(video_fps)

        tmp_dir = tempfile.mkdtemp(prefix="tracker_cfg_")
        tmp_path = os.path.join(tmp_dir, base)

        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        return tmp_path

    # -------------------------------------------------------------------------
    # THE FUNCTION YOU ASKED TO REWRITE (FULL)
    # -------------------------------------------------------------------------

    def tracking_mode_threadsafe(
        self,
        input_video_path: str,
        video_fps: int,
        bbox_mode: bool,
        seg_mode: bool,
        bbox_csv_out: Optional[str] = None,
        seg_csv_out: Optional[str] = None,
        bbox_model: YOLO | None = None,
        seg_model: YOLO | None = None,
        flush_every_n_frames: int = 30,
        # Progress UI / logging
        job_label: str = "",
        show_frame_pbar: bool = True,
        tqdm_position: int = 1,
        postfix_every_n: int = 30,
        # Output semantics
        remap_track_ids_per_segment: bool = True,
    ) -> None:
        """
        Thread-safe alternative to legacy tracking_mode():
          - reads frames via OpenCV
          - calls Ultralytics track(frame, ...) per frame
          - writes CSVs directly in the SAME schema you used before
          - does NOT use runs/ paths, and does NOT create per-frame txt artifacts

        IMPORTANT:
          - Each segment/clip is tracked independently.
          - Tracker reset per clip is enforced via persist=False on the first frame.
          - By default, we REMAP track IDs within each clip so unique-id starts from 1
            (this avoids confusing "starts from 11" when the underlying tracker keeps
             incrementing IDs across multiple clips processed by the same thread).
        """

        if bbox_mode and (bbox_csv_out is None):
            raise ValueError("bbox_mode=True requires bbox_csv_out")
        if seg_mode and (seg_csv_out is None):
            raise ValueError("seg_mode=True requires seg_csv_out")

        # Use thread-local models if not provided
        if bbox_model is None or (seg_mode and seg_model is None):
            bbox_m, seg_m = self.get_thread_models(bbox_mode, seg_mode)
            if bbox_model is None:
                bbox_model = bbox_m
            if seg_model is None:
                seg_model = seg_m

        if bbox_mode and bbox_model is None:
            raise ValueError("bbox_mode=True but bbox_model is None")
        if seg_mode and seg_model is None:
            raise ValueError("seg_mode=True but seg_model is None")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        bbox_tracker_eff = self.make_tracker_config(self.bbox_tracker, video_fps) if bbox_mode else self.bbox_tracker
        seg_tracker_eff = self.make_tracker_config(self.seg_tracker, video_fps) if seg_mode else self.seg_tracker

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            total_frames = 0  # tqdm will run without a known total

        bbox_header = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]
        seg_header = ["yolo-id", "mask-polygon", "unique-id", "confidence", "frame-count"]

        bbox_buf: list[list] = []
        seg_buf: list[list] = []
        frame_count = 0

        # Optional per-clip ID remapping (avoids 11/12/13 starts, etc.)
        bbox_id_map: dict[int, int] = {}
        seg_id_map: dict[int, int] = {}
        bbox_next_id = 1
        seg_next_id = 1

        def _map_id(raw_id: int, id_map: dict[int, int], next_id_holder: list[int]) -> int:
            # raw_id could be -1 if tracker didn't provide one
            if raw_id is None or raw_id < 0:
                return -1
            if raw_id in id_map:
                return id_map[raw_id]
            nid = next_id_holder[0]
            id_map[raw_id] = nid
            next_id_holder[0] += 1
            return nid

        bbox_next = [bbox_next_id]
        seg_next = [seg_next_id]

        # Per-frame tqdm
        pbar = None
        if show_frame_pbar:
            desc = f"frames: {job_label}" if job_label else "frames"
            pbar = tqdm(
                total=total_frames if total_frames > 0 else None,
                desc=desc,
                unit="f",
                dynamic_ncols=True,
                position=tqdm_position,
                leave=False,
            )

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_count += 1
                persist_flag = frame_count > 1  # reset per clip; persist within clip

                # ---------------- SEG ----------------
                if seg_mode:
                    seg_results = seg_model.track(  # type: ignore
                        frame,
                        tracker=seg_tracker_eff,
                        persist=persist_flag,
                        conf=self.confidence,
                        verbose=False,
                        device=device,
                        save=False,
                        save_txt=False,
                        show=False,
                    )
                    r = seg_results[0]
                    boxes = r.boxes
                    masks = getattr(r, "masks", None)

                    if boxes is not None and boxes.xywhn is not None and boxes.xywhn.size(0) > 0:
                        n = int(boxes.xywhn.size(0))
                        cls_list = boxes.cls.int().cpu().tolist()
                        raw_id_list = (
                            boxes.id.int().cpu().tolist()
                            if getattr(boxes, "id", None) is not None
                            else [-1] * n
                        )
                        conf_list = (
                            boxes.conf.cpu().tolist()
                            if getattr(boxes, "conf", None) is not None
                            else [math.nan] * n
                        )

                        if masks is not None and getattr(masks, "xyn", None) is not None:
                            polys = masks.xyn
                            m = min(len(polys), n)
                            for i in range(m):
                                poly = polys[i]
                                flat = []
                                for x, y in poly:
                                    flat.append(str(float(x)))
                                    flat.append(str(float(y)))

                                raw_tid = int(raw_id_list[i]) if i < len(raw_id_list) else -1
                                tid = (
                                    _map_id(raw_tid, seg_id_map, seg_next)
                                    if remap_track_ids_per_segment
                                    else raw_tid
                                )

                                seg_buf.append([
                                    int(cls_list[i]),
                                    " ".join(flat),
                                    int(tid),
                                    float(conf_list[i]),
                                    frame_count,
                                ])

                # ---------------- BBOX ----------------
                if bbox_mode:
                    bbox_results = bbox_model.track(  # type: ignore
                        frame,
                        tracker=bbox_tracker_eff,
                        persist=persist_flag,
                        conf=self.confidence,
                        verbose=False,
                        device=device,
                        save=False,
                        save_txt=False,
                        show=False,
                    )
                    r = bbox_results[0]
                    boxes = r.boxes

                    if boxes is not None and boxes.xywhn is not None and boxes.xywhn.size(0) > 0:
                        xywhn = boxes.xywhn.cpu().tolist()
                        cls_list = boxes.cls.int().cpu().tolist()
                        raw_id_list = (
                            boxes.id.int().cpu().tolist()
                            if getattr(boxes, "id", None) is not None
                            else [-1] * len(xywhn)
                        )
                        conf_list = (
                            boxes.conf.cpu().tolist()
                            if getattr(boxes, "conf", None) is not None
                            else [math.nan] * len(xywhn)
                        )

                        for j, ((x, y, w, h), c, raw_tid, confv) in enumerate(zip(xywhn, cls_list, raw_id_list, conf_list)):  # noqa: E501
                            raw_tid = int(raw_tid) if raw_tid is not None else -1
                            tid = (
                                _map_id(raw_tid, bbox_id_map, bbox_next)
                                if remap_track_ids_per_segment
                                else raw_tid
                            )

                            bbox_buf.append([
                                int(c),
                                float(x),
                                float(y),
                                float(w),
                                float(h),
                                int(tid),
                                float(confv),
                                frame_count,
                            ])

                # Flush periodically
                if frame_count % flush_every_n_frames == 0:
                    if bbox_mode and bbox_csv_out:
                        self._write_rows_csv(bbox_csv_out, bbox_header, bbox_buf)
                        bbox_buf.clear()
                    if seg_mode and seg_csv_out:
                        self._write_rows_csv(seg_csv_out, seg_header, seg_buf)
                        seg_buf.clear()

                # tqdm update
                if pbar is not None:
                    pbar.update(1)
                    if postfix_every_n and (frame_count % postfix_every_n == 0):
                        pbar.set_postfix({
                            "f": frame_count,
                            "bbox_rows": (0 if not bbox_mode else len(bbox_buf)),
                            "seg_rows": (0 if not seg_mode else len(seg_buf)),
                        })

            # final flush
            if bbox_mode and bbox_csv_out:
                self._write_rows_csv(bbox_csv_out, bbox_header, bbox_buf)
            if seg_mode and seg_csv_out:
                self._write_rows_csv(seg_csv_out, seg_header, seg_buf)

        finally:
            cap.release()
            if pbar is not None:
                pbar.close()

            # cleanup temp tracker configs (only when we created a temp dir)
            for p in (bbox_tracker_eff, seg_tracker_eff):
                if isinstance(p, str) and "tracker_cfg_" in p:
                    try:
                        shutil.rmtree(os.path.dirname(p), ignore_errors=True)
                    except Exception:
                        pass
