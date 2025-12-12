# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

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
import pycountry
import requests
import torch
import world_bank_data as wb
import yaml
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

# Consts (kept for legacy path)
LINE_THICKNESS = 1
RENDER = False
SHOW_LABELS = False
SHOW_CONF = False

UPGRADE_LOG_FILE = "upgrade_log.json"


class Youtube_Helper:
    """
    Helper class for:
      - Video acquisition (HTTP file server first, then YouTube fallback)
      - Trimming and video metadata
      - Mapping enrichment (ISO3, population, mortality, gini, upload_date)
      - Thread-safe YOLO tracking/segmentation CSV generation
    """

    _TLS = threading.local()

    def __init__(self, video_title: Optional[str] = None):
        self.tracking_model = common.get_configs("tracking_model")
        self.segment_model = common.get_configs("segment_model")

        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.seg_tracker = common.get_configs("seg_tracker")

        # Confidence: keep old default behavior; prefer config if available
        try:
            self.confidence = float(common.get_configs("confidence"))
        except Exception:
            self.confidence = 0.0

        try:
            self.track_buffer_sec = float(common.get_configs("track_buffer_sec"))
        except Exception:
            self.track_buffer_sec = 1.0

        # Legacy flags (kept for compatibility)
        self.display_frame_tracking = common.get_configs("display_frame_tracking")
        self.display_frame_segmentation = common.get_configs("display_frame_segmentation")
        self.save_annoted_img = common.get_configs("save_annoted_img")
        self.save_tracked_img = common.get_configs("save_tracked_img")
        self.delete_labels = common.get_configs("delete_labels")
        self.delete_frames = common.get_configs("delete_frames")

        self.update_package = common.get_configs("update_package")
        self.need_authentication = common.get_configs("need_authentication")
        self.client = common.get_configs("client")

        self.video_title = video_title or ""

        # Mapping path (do not keep a DF here as authoritative state)
        self.mapping_path = common.get_configs("mapping")

    # -------------------------------------------------------------------------
    # Utility: video title
    # -------------------------------------------------------------------------
    def set_video_title(self, title: str) -> None:
        self.video_title = title

    # -------------------------------------------------------------------------
    # Utility: folder rename/delete
    # -------------------------------------------------------------------------
    def rename_folder(self, old_name: str, new_name: str) -> None:
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            logger.error(f"Error: Folder '{old_name}' not found.")
        except FileExistsError:
            logger.error(f"Error: Folder '{new_name}' already exists.")

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

    # -------------------------------------------------------------------------
    # Package upgrade logging
    # -------------------------------------------------------------------------
    def load_upgrade_log(self) -> dict[str, str]:
        if not os.path.exists(UPGRADE_LOG_FILE):
            return {}
        try:
            with open(UPGRADE_LOG_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return {}

    def save_upgrade_log(self, log_data: dict[str, str]) -> None:
        with open(UPGRADE_LOG_FILE, "w", encoding="utf-8") as file:
            json.dump(log_data, file)

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
    # Safe copy to SSD
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
                except (OSError, AttributeError):
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
    # HTTP file-server ("FTP") download (kept name for compatibility)
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

            # 1) direct URLs
            for alias in aliases:
                direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
                r = fetch(direct_url, stream=True)
                if r is None:
                    continue

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                # avoid overwriting
                if os.path.exists(local_path):
                    stem, suf = os.path.splitext(local_path)
                    i = 1
                    while os.path.exists(f"{stem} ({i}){suf}"):
                        i += 1
                    local_path = f"{stem} ({i}){suf}"
                    logger.warning(f"File exists, saving as: {local_path}")

                try:
                    total = int(r.headers.get("content-length", 0)) or None
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

                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path) or 0.0)
                    resolution = self.get_video_resolution_label(local_path)
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename, resolution, fps

            # 2) crawl /browse
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

                return None

            for alias in aliases:
                start_url = urljoin(base, f"v/{alias}/browse")
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
                    fps = float(self.get_video_fps(local_path) or 0.0)
                    resolution = self.get_video_resolution_label(local_path)
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename, resolution, fps

            logger.warning(f"File '{filename_with_ext}' not found in any alias.")
            return None

    # -------------------------------------------------------------------------
    # YouTube download with resolution preference (pytubefix -> yt_dlp fallback)
    # -------------------------------------------------------------------------
    def download_video_with_resolution(self, vid: str, resolutions: list[str] | None = None, output_path: str = "."):
        if resolutions is None:
            resolutions = ["720p", "480p", "360p", "144p"]

        # pytubefix path
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
                youtube_object = YouTube(
                    youtube_url,
                    self.client,
                    on_progress_callback=on_progress,
                )

            selected_stream = None
            selected_resolution = None

            for resolution in resolutions:
                video_streams = youtube_object.streams.filter(res=resolution)
                if video_streams:
                    selected_resolution = resolution
                    selected_stream = video_streams.first() if hasattr(video_streams, "first") else video_streams[0]
                    break

            # fallback: highest available <=720p mp4 (progressive preferred)
            if not selected_stream:
                def _height_from_res(res_str: str) -> int:
                    if not res_str:
                        return -1
                    m = re.search(r"(\d{3,4})p", res_str)
                    return int(m.group(1)) if m else -1

                progressive = []
                adaptive = []
                for s in youtube_object.streams:
                    res_attr = getattr(s, "resolution", None) or getattr(s, "res", None)
                    h = _height_from_res(str(res_attr))
                    if 0 < h <= 720:
                        mime = (getattr(s, "mime_type", "") or "").lower()
                        if "mp4" not in mime:
                            continue
                        if getattr(s, "is_progressive", False):
                            progressive.append((h, s))
                        else:
                            adaptive.append((h, s))

                chosen = None
                if progressive:
                    chosen = max(progressive, key=lambda t: t[0])[1]
                elif adaptive:
                    chosen = max(adaptive, key=lambda t: t[0])[1]

                if not chosen:
                    logger.error(f"{vid}: no usable stream <= 720p.")
                    return None

                selected_stream = chosen
                selected_resolution = getattr(chosen, "resolution", None) or getattr(chosen, "res", None)

            video_file_path = os.path.join(output_path, f"{vid}.mp4")
            logger.info(f"{vid}: download started with pytubefix (res={selected_resolution}).")

            selected_stream.download(output_path, filename=f"{vid}.mp4")
            self.video_title = youtube_object.title
            fps = self.get_video_fps(video_file_path)
            return video_file_path, vid, str(selected_resolution), fps

        except Exception as e:
            logger.error(f"{vid}: pytubefix download method failed: {e}")
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
                raise RuntimeError(f"{vid}: no stream available via yt_dlp in requested resolutions.")

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

            logger.info(f"{vid}: download started with yt_dlp (res={selected_resolution}).")
            with yt_dlp.YoutubeDL(download_opts) as ydl:  # type: ignore
                ydl.download([youtube_url])

            video_file_path = os.path.join(output_path, f"{vid}.mp4")
            self.video_title = str(info_dict.get("title") or vid)
            fps = self.get_video_fps(video_file_path)
            return video_file_path, vid, str(selected_resolution), fps

        except Exception as e:
            logger.error(f"{vid}: yt_dlp download method failed: {e}.")
            return None

    # -------------------------------------------------------------------------
    # Video metadata + trimming
    # -------------------------------------------------------------------------
    def get_video_fps(self, video_file_path: str) -> Optional[float]:
        try:
            video = cv2.VideoCapture(video_file_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            video.release()
            return round(float(fps), 0)
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
            144: "144p",
            240: "240p",
            360: "360p",
            480: "480p",
            720: "720p",
            1080: "1080p",
            1440: "1440p",
            2160: "2160p",
        }
        if height in labels:
            return labels[height]

        tolerance = 8
        closest = min(labels.keys(), key=lambda h: abs(h - height))
        if abs(closest - height) <= tolerance:
            return labels[closest]

        return f"{height}p"

    def trim_video(self, input_path: str, output_path: str, start_time: float, end_time: float) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)
        # keep behavior; suppress verbose output
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        video_clip.close()

    # -------------------------------------------------------------------------
    # Compression
    # -------------------------------------------------------------------------
    def detect_gpu(self) -> Optional[str]:
        try:
            nvidia_check = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if nvidia_check.returncode == 0:
                return "hevc_nvenc"

            intel_check = subprocess.run(["vainfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "Intel" in intel_check.stdout:
                return "hevc_qsv"
        except FileNotFoundError:
            pass
        return None

    def extract_youtube_id(self, file_path: str) -> str:
        filename = os.path.basename(file_path)
        youtube_id, _ext = os.path.splitext(filename)
        if not youtube_id or len(youtube_id) < 5:
            raise ValueError("Invalid YouTube ID extracted.")
        return youtube_id

    def compress_video(self, input_path: str, codec: str = "libx265", preset: str = "slow", crf: int = 17) -> None:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        filename = os.path.basename(input_path)
        tmp_out = os.path.join(common.root_dir, f"__tmp_compress_{filename}")

        codec_hw = self.detect_gpu()
        if codec_hw:
            codec = codec_hw

        # Correct ffmpeg arg order: output path should be last
        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", codec,
            "-preset", preset,
            "-crf", str(crf),
            "-progress", "pipe:1",
            "-nostats",
            tmp_out,
        ]

        try:
            video_id = self.extract_youtube_id(input_path)
            logger.info(
                f"Started compression of {video_id} with {codec}. Current size={os.path.getsize(input_path)}."
            )
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.info(
                f"Finished compression of {video_id} with {codec}. New size={os.path.getsize(tmp_out)}."
            )
            shutil.move(tmp_out, input_path)
        except Exception as e:
            if os.path.exists(tmp_out):
                try:
                    os.remove(tmp_out)
                except Exception:
                    pass
            logger.error(f"Video compression failed: {e}. Using uncompressed file.")

    # -------------------------------------------------------------------------
    # Cleanup helpers
    # -------------------------------------------------------------------------
    def delete_youtube_mod_videos(self, folders: list[str]) -> None:
        # supports both legacy "{id}_mod.mp4" and new unique variants "{id}_*_mod.mp4"
        patterns = [
            re.compile(r"^[A-Za-z0-9_-]{11}_mod\.mp4$"),
            re.compile(r"^[A-Za-z0-9_-]{11}_.+_mod\.mp4$"),
        ]
        for folder in folders:
            if not os.path.exists(folder):
                logger.info(f"Skipping missing folder: {folder}")
                continue
            for filename in os.listdir(folder):
                if any(p.match(filename) for p in patterns):
                    file_path = os.path.join(folder, filename)
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted: {file_path}")
                    except Exception as e:
                        logger.info(f"Failed to delete {file_path}: {e}")

    # -------------------------------------------------------------------------
    # Mapping enrichment: ISO3 / population / mortality / gini / upload date
    # -------------------------------------------------------------------------
    def get_iso_alpha_3(self, country_name: str, existing_iso: Optional[str]) -> Optional[str]:
        try:
            return pycountry.countries.lookup(country_name).alpha_3
        except LookupError:
            if country_name.strip().upper() == "KOSOVO":
                return "XKX"
            return existing_iso if existing_iso else None

    def get_latest_population(self) -> pd.DataFrame:
        indicator = "SP.POP.TOTL"
        population_data = wb.get_series(indicator, id_or_value="id", mrv=1)
        population_df = population_data.reset_index()
        population_df = population_df.rename(
            columns={
                population_df.columns[0]: "iso3",
                population_df.columns[2]: "Year",
                population_df.columns[3]: "Population",
            }
        )
        population_df["Population"] = population_df["Population"] / 1000
        return population_df

    def update_population_in_csv(self, data: pd.DataFrame) -> None:
        if "iso3" not in data.columns:
            raise KeyError("The CSV file does not have an 'iso3' column.")

        if "population_country" not in data.columns:
            data["population_country"] = None

        latest_population = self.get_latest_population()
        population_dict = dict(zip(latest_population["iso3"], latest_population["Population"]))

        for index, row in data.iterrows():
            iso3 = row["iso3"]
            data.at[index, "population_country"] = population_dict.get(iso3, None)

        # IMPORTANT FIX: write to path, not self.mapping DF
        data.to_csv(self.mapping_path, index=False)
        logger.info("Mapping file updated successfully with country population.")

    # -------------------------------------------------------------------------
    # Thread-safe YOLO tracking/segmentation (NEW)
    # -------------------------------------------------------------------------
    def get_thread_models(self, bbox_mode: bool, seg_mode: bool) -> tuple[Optional[YOLO], Optional[YOLO]]:
        # bbox model
        if bbox_mode:
            if not hasattr(self._TLS, "bbox_model") or self._TLS.bbox_model is None:
                logger.info(f"[models] Loading bbox model in thread={threading.current_thread().name}: {self.tracking_model}")
                self._TLS.bbox_model = YOLO(self.tracking_model)
        else:
            if not hasattr(self._TLS, "bbox_model"):
                self._TLS.bbox_model = None

        # seg model
        if seg_mode:
            if not hasattr(self._TLS, "seg_model") or self._TLS.seg_model is None:
                logger.info(f"[models] Loading seg model in thread={threading.current_thread().name}: {self.segment_model}")
                self._TLS.seg_model = YOLO(self.segment_model)
        else:
            if not hasattr(self._TLS, "seg_model"):
                self._TLS.seg_model = None

        return self._TLS.bbox_model, self._TLS.seg_model

    def make_tracker_config(self, tracker_path: str, video_fps: int) -> str:
        if not isinstance(tracker_path, str) or not tracker_path.endswith(".yaml"):
            return tracker_path

        # Only edit if it is an actual local file
        if not os.path.isfile(tracker_path):
            return tracker_path

        base = os.path.basename(tracker_path)
        if base not in ("bbox_custom_tracker.yaml", "seg_custom_tracker.yaml"):
            return tracker_path

        # Convert seconds to frames (Ultralytics tracker expects frames)
        try:
            track_buffer_sec = float(common.get_configs("track_buffer_sec"))
        except Exception:
            track_buffer_sec = 2.0

        track_buffer_frames = int(round(track_buffer_sec * float(video_fps)))

        with open(tracker_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        cfg["track_buffer"] = track_buffer_frames

        tmp_dir = tempfile.mkdtemp(prefix="tracker_cfg_")
        tmp_path = os.path.join(tmp_dir, base)

        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        return tmp_path

    @staticmethod
    def _write_rows_csv(path: str, header: list[str], rows: list[list]) -> None:
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerows(rows)

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
        # ---- optional tqdm controls (safe defaults) ----
        show_frame_pbar: bool = False,
        tqdm_position: int = 1,
        job_label: str = "",
        postfix_every_n: int = 30,
    ) -> None:
        """
        Thread-safe alternative to legacy tracking_mode():
          - reads frames via OpenCV
          - calls Ultralytics track(frame, ...) per frame
          - writes CSVs directly in the SAME schema you used before
          - does NOT use runs/ paths, and does NOT create per-frame txt artifacts

        IMPORTANT: tracker reset per segment is enforced via persist=False on first frame.
        """

        if bbox_mode and (bbox_csv_out is None):
            raise ValueError("bbox_mode=True requires bbox_csv_out")
        if seg_mode and (seg_csv_out is None):
            raise ValueError("seg_mode=True requires seg_csv_out")

        # If caller didn't provide models, use thread-local ones
        if bbox_model is None or (seg_mode and seg_model is None):
            bbox_m, seg_m = self.get_thread_models(bbox_mode=bbox_mode, seg_mode=seg_mode)
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

        bbox_header = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]
        seg_header = ["yolo-id", "mask-polygon", "unique-id", "confidence", "frame-count"]

        bbox_buf: list[list] = []
        seg_buf: list[list] = []
        frame_count = 0

        pbar = None
        t0 = time.time()

        try:
            if show_frame_pbar:
                pbar = tqdm(
                    total=total_frames if total_frames > 0 else None,
                    desc=f"{job_label} frames" if job_label else "frames",
                    unit="frame",
                    position=tqdm_position,
                    leave=False,
                    dynamic_ncols=True,
                )

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_count += 1
                persist_flag = frame_count > 1  # reset per clip; persist within clip

                # ---------------- SEG ----------------
                if seg_mode:
                    seg_results = seg_model.track(
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
                        id_list = boxes.id.int().cpu().tolist() if getattr(boxes, "id", None) is not None else [-1] * n
                        conf_list = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [math.nan] * n

                        # masks.xyn is a list of polygons (normalized points)
                        if masks is not None and getattr(masks, "xyn", None) is not None:
                            polys = masks.xyn
                            m = min(len(polys), n)
                            for i in range(m):
                                poly = polys[i]
                                flat = []
                                for x, y in poly:
                                    flat.append(str(float(x)))
                                    flat.append(str(float(y)))
                                seg_buf.append([
                                    int(cls_list[i]),
                                    " ".join(flat),
                                    int(id_list[i]),
                                    float(conf_list[i]),
                                    frame_count
                                ])

                # ---------------- BBOX ----------------
                if bbox_mode:
                    bbox_results = bbox_model.track(
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
                        id_list = boxes.id.int().cpu().tolist() if getattr(boxes, "id", None) is not None else [-1] * len(xywhn)
                        conf_list = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [math.nan] * len(xywhn)

                        for (x, y, w, h), c, tid, confv in zip(xywhn, cls_list, id_list, conf_list):
                            bbox_buf.append([int(c), float(x), float(y), float(w), float(h), int(tid), float(confv), frame_count])

                # ---------------- flush ----------------
                if frame_count % flush_every_n_frames == 0:
                    if bbox_mode and bbox_csv_out:
                        self._write_rows_csv(bbox_csv_out, bbox_header, bbox_buf)
                        bbox_buf.clear()
                    if seg_mode and seg_csv_out:
                        self._write_rows_csv(seg_csv_out, seg_header, seg_buf)
                        seg_buf.clear()

                # ---------------- tqdm ----------------
                if pbar:
                    pbar.update(1)
                    if postfix_every_n and (frame_count % postfix_every_n == 0):
                        elapsed = time.time() - t0
                        eff_fps = (frame_count / elapsed) if elapsed > 0 else 0.0
                        pbar.set_postfix_str(f"yolo_fps={eff_fps:.2f}")

            # final flush
            if bbox_mode and bbox_csv_out:
                self._write_rows_csv(bbox_csv_out, bbox_header, bbox_buf)
            if seg_mode and seg_csv_out:
                self._write_rows_csv(seg_csv_out, seg_header, seg_buf)

        finally:
            cap.release()
            if pbar:
                pbar.close()

            # cleanup temp tracker configs
            for p in (bbox_tracker_eff, seg_tracker_eff):
                if isinstance(p, str) and "tracker_cfg_" in p:
                    try:
                        shutil.rmtree(os.path.dirname(p), ignore_errors=True)
                    except Exception:
                        pass

    # -------------------------------------------------------------------------
    # Legacy tracking_mode (kept for compatibility; NOT thread-safe)
    # -------------------------------------------------------------------------
    def update_track_buffer_in_yaml(self, yaml_path: str, video_fps: int) -> None:
        """
        Legacy in-place updater (NOT thread-safe). Prefer make_tracker_config() instead.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["track_buffer"] = float(self.track_buffer_sec) * float(video_fps)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def tracking_mode(self, *args, **kwargs):
        """
        LEGACY METHOD (not thread-safe):
        retained so older runners don't break.

        In the multithreaded pipeline, call tracking_mode_threadsafe() instead.
        """
        raise RuntimeError(
            "tracking_mode() is legacy and not thread-safe. "
            "Use tracking_mode_threadsafe() in the multithreaded pipeline."
        )
