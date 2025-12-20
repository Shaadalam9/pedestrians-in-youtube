import math
import os
from tqdm import tqdm
from typing import Tuple, List, Optional, Set, Any
import subprocess
import shutil
import time
import cv2
import re
import requests
import pathlib
import datetime
from types import SimpleNamespace
from bs4 import BeautifulSoup
import yt_dlp
import common
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.VideoFileClip import VideoFileClip
from urllib.parse import urljoin, urlparse
from custom_logger import CustomLogger
from parsing_utils import ParsingUtils
from config_utils import ConfigUtils
from maintenence.maintenence import Maintenance

parsing_utils = ParsingUtils()
config_utils = ConfigUtils()
maintenance_class = Maintenance()
logger = CustomLogger(__name__)


class VideoIO:
    def __init__(self) -> None:
        # Snellius/HPC flag(s)
        self.snellius_mode = bool(config_utils._safe_get_config("snellius_mode", False))
        # Default behavior on HPC: do not download from internet/FTP on compute nodes
        self.snellius_disable_downloads = bool(config_utils._safe_get_config("snellius_disable_downloads",
                                                                             True if self.snellius_mode else False))
        self.update_package = bool(config_utils._safe_get_config("update_package", False))
        self.need_authentication = bool(config_utils._safe_get_config("need_authentication", False))

    def set_video_title(self, title: str) -> None:
        self.video_title = title

    def _fps_is_bad(self, fps) -> bool:
        return fps is None or fps == 0 or (isinstance(fps, float) and math.isnan(fps))

    def _copy_to_ssd_if_needed(self, base_video_path: str, internal_ssd: str, vid: str) -> str:
        """
        Preserves the prior 'copy_video_safe' behavior.
        Ensures the file is on SSD and returns the SSD path.
        """
        if os.path.dirname(base_video_path) == internal_ssd:
            return base_video_path

        out = self.copy_video_safe(base_video_path, internal_ssd, vid)
        logger.debug(f"Copied to {out}.")
        return os.path.join(internal_ssd, f"{vid}.mp4")

    def _trim_video_with_progress(self, input_path: str, output_path: str, start_time: int,
                                  end_time: int, job_label: str, tqdm_position: int):
        """
        Trim with a tqdm progress bar using ffmpeg -progress pipe:1.

        Strategy:
          1) Try stream-copy (fast). If progress doesn't move (common), retry with re-encode.
          2) Re-encode guarantees progress updates but is slower.
        """
        duration = max(0.0, float(end_time) - float(start_time))
        if duration <= 0.0:
            self.trim_video(input_path, output_path, start_time, end_time)
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        def _run_ffmpeg(cmd: List[str]) -> float:
            pbar = tqdm(
                total=duration,
                desc=f"trim: {job_label}",
                unit="s",
                dynamic_ncols=True,
                position=tqdm_position,
                leave=False,  # shows while trimming; then yields same line to frames pbar
            )
            last_sec = 0.0
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, universal_newlines=True)

                if proc.stdout is not None:
                    for line in proc.stdout:
                        line = line.strip()
                        if not line:
                            continue

                        if line.startswith("out_time_ms="):
                            try:
                                ms = int(line.split("=", 1)[1].strip())
                                cur = ms / 1_000_000.0
                            except Exception:
                                continue
                            cur = min(cur, duration)
                            if cur > last_sec:
                                pbar.update(cur - last_sec)
                                last_sec = cur

                        elif line.startswith("out_time="):
                            cur = parsing_utils._hms_to_seconds(line.split("=", 1)[1].strip())
                            if cur is None:
                                continue
                            cur = min(cur, duration)
                            if cur > last_sec:
                                pbar.update(cur - last_sec)
                                last_sec = cur

                        elif line.startswith("progress=") and line.endswith("end"):
                            break

                rc = proc.wait()
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)

                if last_sec < duration:
                    pbar.update(duration - last_sec)

                return last_sec

            finally:
                try:
                    pbar.close()
                except Exception:
                    pass

        # 1) Fast path: stream copy
        cmd_copy = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start_time), "-to", str(end_time),
                    "-i", input_path, "-c", "copy", "-avoid_negative_ts", "1", "-movflags", "+faststart", "-progress",
                    "pipe:1", "-nostats", output_path]

        try:
            progressed = _run_ffmpeg(cmd_copy)

            # If no visible progress, retry with re-encode for a real bar
            if progressed < 1.0:
                logger.info(f"[trim] no progress from stream-copy; retrying with re-encode for: {job_label}")

                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                except Exception:
                    pass

                cmd_reencode = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start_time), "-to",
                                str(end_time), "-i", input_path, "-c:v", "libx264", "-preset", "veryfast", "-crf",
                                "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-progress", "pipe:1",
                                "-nostats", output_path]
                _run_ffmpeg(cmd_reencode)

        except Exception as e:
            logger.warning(f"[trim] ffmpeg trim failed; falling back to helper.trim_video(). reason={e!r}")
            self.trim_video(input_path, output_path, start_time, end_time)

    def _ensure_video_available(self, vid: str, config: SimpleNamespace, secret: SimpleNamespace, output_path: str,
                                video_paths: List[str]) -> Tuple[str, str, str, int, bool]:
        """
        Preserves your prior retrieval logic.
        Returns: (base_video_path, video_title, resolution, video_fps, ftp_download)

        Snellius/HPC addition:
          - if snellius_disable_youtube_fallback=True (and snellius_mode=True),
            never attempt YouTube downloads. Only cached/FTP are allowed.
        """
        ftp_download = False
        resolution = "unknown"
        video_title = vid

        base_video_path = os.path.join(output_path, f"{vid}.mp4")

        disable_yt = bool(getattr(config, "snellius_mode", False)) and bool(
            getattr(config, "snellius_disable_youtube_fallback", False)
        )

        if config.external_ssd:
            existing_path = os.path.join(output_path, f"{vid}.mp4")

            if os.path.exists(existing_path):
                tmp_dir = os.path.join(output_path, "__tmp_dl")
                os.makedirs(tmp_dir, exist_ok=True)

                logger.info(f"{vid}: SSD copy exists; attempting FTP refresh into temp={tmp_dir}.")
                tmp_result = self.download_videos_from_ftp(
                    filename=vid,
                    base_url=config.ftp_server,
                    out_dir=tmp_dir,
                    username=secret.ftp_username,
                    password=secret.ftp_password,
                )

                if tmp_result:
                    tmp_video_path, video_title, resolution, video_fps = tmp_result
                    if self._fps_is_bad(video_fps):
                        logger.warning(f"{vid}: invalid FPS in refreshed file; keeping existing SSD copy.")
                        try:
                            os.remove(tmp_video_path)
                        except Exception:
                            pass
                        video_fps2 = self.get_video_fps(existing_path)
                        if self._fps_is_bad(video_fps2):
                            raise RuntimeError(f"{vid}: existing SSD copy also has invalid FPS.")
                        self.set_video_title(video_title)
                        return existing_path, video_title, resolution, int(video_fps2), False  # type: ignore

                    try:
                        os.remove(existing_path)
                    except FileNotFoundError:
                        pass

                    final_path = os.path.join(output_path, f"{vid}.mp4")
                    shutil.move(tmp_video_path, final_path)
                    ftp_download = True
                    self.set_video_title(video_title)

                    logger.info(f"{vid}: refreshed from FTP and replaced SSD copy at {final_path}.")
                    return final_path, video_title, resolution, int(video_fps), ftp_download

                logger.info(f"{vid}: FTP not available; using existing SSD copy.")
                self.set_video_title(video_title)
                video_fps = self.get_video_fps(existing_path)

                if self._fps_is_bad(video_fps):
                    if disable_yt:
                        raise RuntimeError(f"{vid}: invalid FPS on SSD copy and YouTube fallback disabled (Snellius).")
                    logger.warning(f"{vid}: invalid FPS on SSD copy; attempting YouTube fallback.")
                    yt_result = self.download_video_with_resolution(vid=vid, output_path=output_path)
                    if not yt_result:
                        raise RuntimeError(f"{vid}: YouTube fallback failed and SSD copy FPS invalid.")
                    video_file_path, video_title, resolution, video_fps = yt_result
                    if self._fps_is_bad(video_fps):
                        raise RuntimeError(f"{vid}: YouTube fallback produced invalid FPS.")
                    self.set_video_title(video_title)
                    return video_file_path, video_title, resolution, int(video_fps), False

                return existing_path, video_title, resolution, int(video_fps), False  # type: ignore

            logger.info(f"{vid}: no SSD copy; attempting FTP download to {output_path}.")
            result = self.download_videos_from_ftp(filename=vid, base_url=config.ftp_server, out_dir=output_path,
                                                   username=secret.ftp_username, password=secret.ftp_password)
            if result:
                ftp_download = True
            if result is None:
                if disable_yt:
                    raise RuntimeError(f"{vid}: FTP not found/failed and YouTube fallback disabled (Snellius).")
                logger.info(f"{vid}: FTP not found/failed; attempting YouTube download.")
                result = self.download_video_with_resolution(vid=vid, output_path=output_path)

            if not result:
                raise RuntimeError(f"{vid}: forced download failed (FTP+fallback).")

            video_file_path, video_title, resolution, video_fps = result
            if self._fps_is_bad(video_fps):
                raise RuntimeError(f"{vid}: invalid video_fps after download.")
            self.set_video_title(video_title)

            logger.info(f"{vid}: downloaded successfully. res={resolution} fps={int(video_fps)} path={video_file_path}")  # noqa: E501
            return video_file_path, video_title, resolution, int(video_fps), ftp_download

        exists_somewhere = any(os.path.exists(os.path.join(path, f"{vid}.mp4")) for path in video_paths)

        if not exists_somewhere:
            logger.info(f"{vid}: not cached; attempting FTP download to {output_path}.")
            result = self.download_videos_from_ftp(filename=vid, base_url=config.ftp_server, out_dir=output_path,
                                                   username=secret.ftp_username, password=secret.ftp_password)
            if result:
                ftp_download = True
            if result is None:
                if disable_yt:
                    raise RuntimeError(f"{vid}: FTP not found/failed and YouTube fallback disabled (Snellius).")
                logger.info(f"{vid}: FTP not found/failed; attempting YouTube download.")
                result = self.download_video_with_resolution(vid=vid, output_path=output_path)

            if result:
                video_file_path, video_title, resolution, video_fps = result
                if self._fps_is_bad(video_fps):
                    raise RuntimeError(f"{vid}: invalid video_fps after download.")
                self.set_video_title(video_title)
                logger.info(
                    f"{vid}: downloaded successfully. res={resolution} fps={int(video_fps)} path={video_file_path}"
                )
                return video_file_path, video_title, resolution, int(video_fps), ftp_download

            if os.path.exists(base_video_path):
                self.set_video_title(video_title)
                video_fps = self.get_video_fps(base_video_path)
                if self._fps_is_bad(video_fps):
                    raise RuntimeError(f"{vid}: invalid FPS on local fallback file.")
                logger.info(f"{vid}: found locally at {base_video_path}. fps={int(video_fps)}")  # type: ignore
                return base_video_path, video_title, resolution, int(video_fps), False  # type: ignore

            raise RuntimeError(f"{vid}: video not found and download failed.")

        existing_folder = next((p for p in video_paths if os.path.exists(os.path.join(p, f"{vid}.mp4"))), None)
        use_folder = existing_folder if existing_folder else video_paths[-1]
        base_video_path = os.path.join(use_folder, f"{vid}.mp4")
        self.set_video_title(video_title)

        video_fps = self.get_video_fps(base_video_path)
        if self._fps_is_bad(video_fps):
            raise RuntimeError(f"{vid}: invalid FPS on cached file.")

        logger.info(f"{vid}: using cached video at {base_video_path}. fps={int(video_fps)}")  # type: ignore
        return base_video_path, video_title, resolution, int(video_fps), False  # type: ignore

    def _wait_for_stable_file(self, src: str, checks: int = 2, interval: float = 0.5, timeout: float = 30) -> bool:
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

    def copy_video_safe(self, base_video_path: str, internal_ssd: str, vid: str, max_attempts: int = 5,
                        backoff: float = 0.6) -> str:
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

        if not self._wait_for_stable_file(base_video_path):
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
    # Download: FTP-like HTTP file server (kept, but can be disabled on Snellius)
    # -------------------------------------------------------------------------

    def download_videos_from_ftp(self, filename: str, base_url: Optional[str] = None, out_dir: str = ".",
                                 username: Optional[str] = None, password: Optional[str] = None,
                                 token: Optional[str] = None, timeout: int = 20, debug: bool = True,
                                 max_pages: int = 500) -> Optional[tuple[str, str, str, float]]:
        if self.snellius_mode and self.snellius_disable_downloads:  # type: ignore
            logger.error(
                f"Snellius mode: downloads disabled (snellius_disable_downloads=True). "
                f"Please stage '{filename}.mp4' to /projects or your configured videos folder."
            )
            return None

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

    def download_video_with_resolution(self, vid: str, resolutions: list[str] = ["720p", "480p", "360p", "144p"],
                                       output_path: str = ".") -> Optional[tuple[str, str, str, float]]:
        if self.snellius_mode and self.snellius_disable_downloads:
            logger.error(
                f"Snellius mode: downloads disabled (snellius_disable_downloads=True). "
                f"Please stage '{vid}.mp4' to /projects or your configured videos folder."
            )
            return None

        try:
            if self.update_package and datetime.datetime.today().weekday() == 0:
                maintenance_class.upgrade_package_if_needed("pytube")
                maintenance_class.upgrade_package_if_needed("pytubefix")

            youtube_url = f"https://www.youtube.com/watch?v={vid}"
            if self.need_authentication:
                youtube_object = YouTube(youtube_url, self.client, use_oauth=True,  # type: ignore
                                         allow_oauth_cache=True, on_progress_callback=on_progress)
            else:
                youtube_object = YouTube(youtube_url, self.client, on_progress_callback=on_progress)  # type: ignore

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
                maintenance_class.upgrade_package_if_needed("yt_dlp")

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
            }

            # Snellius/HPC: don't assume a browser exists for cookies extraction
            if not self.snellius_mode:
                download_opts["cookiesfrombrowser"] = ("chrome",)

            logger.info(f"{vid}: download in {selected_resolution} started with yt_dlp.")
            with yt_dlp.YoutubeDL(download_opts) as ydl:  # type: ignore
                ydl.download([youtube_url])

            video_file_path = os.path.join(output_path, f"{vid}.mp4")
            self.video_title = info_dict.get("title")
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
            720: "720p", 1080: "1080p", 1440: "1440p", 2160: "2160p",
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