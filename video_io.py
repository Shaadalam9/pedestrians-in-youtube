"""Video I/O utilities for retrieving, validating, and trimming MP4 assets.

This module centralizes logic for:
- Locating cached videos across configured search paths.
- Downloading videos via an FTP-like HTTP file server and (optionally) YouTube.
- Performing safe file copies to faster local storage (e.g., internal SSD).
- Trimming clips with ffmpeg while reporting progress via tqdm.

Operational notes:
- The module supports HPC/Snellius execution constraints via configuration flags that
  disable network downloads on compute nodes.
- Video metadata (FPS and resolution label) is validated to prevent downstream failures.

External dependencies:
- ffmpeg must be available on PATH for progress-aware trimming.
- OpenCV (cv2) is used for basic video metadata extraction.
- requests + BeautifulSoup are used for file-server crawling downloads.

The code intentionally favors robust fallbacks and defensive checks, as video pipelines
can be sensitive to partial/corrupt downloads and inconsistent metadata.
"""

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

# Shared helpers (instantiated once to avoid repeated setup/overhead).
parsing_utils = ParsingUtils()
config_utils = ConfigUtils()
maintenance_class = Maintenance()

# Module-level logger bound to this namespace for consistent structured logging.
logger = CustomLogger(__name__)


class VideoIO:
    """High-level interface for video retrieval, caching, metadata, and trimming.

    This class encapsulates:
    - Environment-aware download policy (e.g., Snellius/HPC restrictions).
    - Retrieval strategy: cached -> FTP-like file server -> YouTube fallback.
    - Safe copying of large binary assets to local SSD.
    - Trim operations (ffmpeg + progress, with MoviePy fallback).

    Instances are lightweight and read operational flags from configuration at init time.
    """

    def __init__(self) -> None:
        """Initialize VideoIO configuration flags from ConfigUtils.

        Configuration keys (best-effort):
          - snellius_mode: enable HPC behavior defaults.
          - snellius_disable_downloads: disallow network downloads if True.
          - update_package: optionally upgrade packages on schedule.
          - need_authentication: whether YouTube access uses OAuth.
          - client: pytubefix client name (e.g., "WEB").

        Notes:
            The code uses ConfigUtils._safe_get_config to avoid hard failures when keys
            are absent; defaults are conservative for HPC usage.
        """
        # Snellius/HPC flag(s)
        self.snellius_mode = bool(config_utils._safe_get_config("snellius_mode", False))
        # Default behavior on HPC: do not download from internet/FTP on compute nodes
        self.snellius_disable_downloads = bool(config_utils._safe_get_config("snellius_disable_downloads",
                                                                             True if self.snellius_mode else False))
        self.update_package = bool(config_utils._safe_get_config("update_package", False))
        self.need_authentication = bool(config_utils._safe_get_config("need_authentication", False))
        self.client = config_utils._safe_get_config("client", "WEB")

    def set_video_title(self, title: str) -> None:
        """Set the human-readable title of the current video.

        Args:
            title: Video title to store on the instance.
        """
        # Stored for downstream reporting and logging; may differ from the video id.
        self.video_title = title

    def _fps_is_bad(self, fps) -> bool:
        """Return True if an FPS value is missing or invalid.

        Args:
            fps: FPS value to validate; may be None, 0, float NaN, or numeric.

        Returns:
            True if fps is None/0/NaN, otherwise False.
        """
        # OpenCV can return 0 or NaN for certain containers; treat those as invalid.
        return fps is None or fps == 0 or (isinstance(fps, float) and math.isnan(fps))

    def _copy_to_ssd_if_needed(self, base_video_path: str, internal_ssd: str, vid: str) -> str:
        """Copy a video file to SSD if it is not already located there.

        This preserves the prior 'copy_video_safe' behavior and ensures the file is
        present on the specified internal SSD directory.

        Args:
            base_video_path: Source path for the video file.
            internal_ssd: Destination directory representing SSD storage.
            vid: Video identifier (used to name the destination file).

        Returns:
            Path to the SSD copy (existing or newly copied).
        """
        # If already on SSD, no-op to avoid redundant I/O.
        if os.path.dirname(base_video_path) == internal_ssd:
            return base_video_path

        # Perform an atomic-ish safe copy into the SSD directory.
        out = self.copy_video_safe(base_video_path, internal_ssd, vid)
        logger.debug(f"Copied to {out}.")
        return os.path.join(internal_ssd, f"{vid}.mp4")

    def _trim_video_with_progress(self, input_path: str, output_path: str, start_time: int,
                                  end_time: int, job_label: str, tqdm_position: int):
        """Trim a video while displaying ffmpeg progress via tqdm.

        Strategy:
            1) Attempt stream-copy (fast, avoids re-encoding). Progress output may be
               sparse or absent depending on container/indexing.
            2) If progress does not advance, retry with re-encode to force progress.
            3) If ffmpeg fails, fall back to the MoviePy-based trim helper.

        Args:
            input_path: Source video file.
            output_path: Destination path for trimmed output.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            job_label: Label used in the tqdm progress description.
            tqdm_position: tqdm bar position, useful for multiple concurrent bars.

        Returns:
            None. Writes a trimmed file to output_path.
        """
        # Compute the trim duration; ffmpeg progress is measured against this.
        duration = max(0.0, float(end_time) - float(start_time))
        if duration <= 0.0:
            # If the interval is invalid, defer to the existing helper behavior.
            self.trim_video(input_path, output_path, start_time, end_time)
            return

        # Ensure destination directory exists before spawning ffmpeg.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        def _run_ffmpeg(cmd: List[str]) -> float:
            """Run ffmpeg with -progress pipe:1 and convert updates into tqdm ticks.

            Args:
                cmd: ffmpeg command list.

            Returns:
                The last progressed timestamp (seconds) observed from ffmpeg output.

            Raises:
                subprocess.CalledProcessError: If ffmpeg exits with non-zero status.
            """
            # A per-job bar; leave=False so it does not clutter output when complete.
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
                # Merge stderr into stdout so we only need one pipe to parse.
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, universal_newlines=True)

                if proc.stdout is not None:
                    for line in proc.stdout:
                        line = line.strip()
                        if not line:
                            continue

                        # ffmpeg progress may be emitted as out_time_ms or out_time.
                        if line.startswith("out_time_ms="):
                            try:
                                ms = int(line.split("=", 1)[1].strip())
                                cur = ms / 1_000_000.0
                            except Exception:
                                # Defensive: ignore malformed progress lines.
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

                        # Termination marker emitted by ffmpeg -progress.
                        elif line.startswith("progress=") and line.endswith("end"):
                            break

                rc = proc.wait()
                if rc != 0:
                    # Preserve standard CalledProcessError semantics for callers.
                    raise subprocess.CalledProcessError(rc, cmd)

                # Ensure the bar reaches 100% even if final progress line is missing.
                if last_sec < duration:
                    pbar.update(duration - last_sec)

                return last_sec

            finally:
                # Always close tqdm to avoid broken output state.
                try:
                    pbar.close()
                except Exception:
                    pass

        # 1) Fast path: stream copy (no re-encode, fastest when it works).
        cmd_copy = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start_time), "-to", str(end_time),
                    "-i", input_path, "-c", "copy", "-avoid_negative_ts", "1", "-movflags", "+faststart", "-progress",
                    "pipe:1", "-nostats", output_path]

        try:
            progressed = _run_ffmpeg(cmd_copy)

            # If no visible progress, retry with re-encode for a real bar.
            if progressed < 1.0:
                logger.info(f"[trim] no progress from stream-copy; retrying with re-encode for: {job_label}")

                # Remove partial output before retry to avoid confusing downstream steps.
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                except Exception:
                    pass

                # 2) Re-encode path: slower but progress and compatibility are better.
                cmd_reencode = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start_time), "-to",
                                str(end_time), "-i", input_path, "-c:v", "libx264", "-preset", "veryfast", "-crf",
                                "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-progress", "pipe:1",
                                "-nostats", output_path]
                _run_ffmpeg(cmd_reencode)

        except Exception as e:
            # Preserve robustness: if ffmpeg path fails, fall back to MoviePy helper.
            logger.warning(f"[trim] ffmpeg trim failed; falling back to helper.trim_video(). reason={e!r}")
            self.trim_video(input_path, output_path, start_time, end_time)

    def _ensure_video_available(self, vid: str, config: SimpleNamespace, secret: SimpleNamespace, output_path: str,
                                video_paths: List[str]) -> Tuple[str, str, str, int, bool]:
        """Ensure a video is available locally and return its metadata.

        Retrieval order (conceptual):
          1) If external_ssd is enabled and a local copy exists, attempt an FTP refresh
             into a temporary directory; validate FPS; replace if valid.
          2) If no SSD copy exists (or external_ssd disabled), attempt download via the
             FTP-like file server.
          3) If FTP fails and YouTube fallback is allowed, attempt YouTube download.
          4) If cached elsewhere in video_paths, use cached file.

        HPC/Snellius behavior:
          - If snellius_mode and snellius_disable_youtube_fallback are enabled, YouTube
            downloads are never attempted; only cached/FTP are allowed.

        Args:
            vid: Video identifier (e.g., YouTube ID / filename stem).
            config: Runtime configuration namespace with fields like external_ssd, ftp_server,
                snellius_mode, snellius_disable_youtube_fallback.
            secret: Secret namespace with FTP credentials (ftp_username, ftp_password).
            output_path: Directory for writing downloads and staged copies.
            video_paths: Candidate directories to search for cached assets.

        Returns:
            A 5-tuple: (base_video_path, video_title, resolution, video_fps, ftp_download)
              - base_video_path: Path to the resolved local MP4 file.
              - video_title: Title (defaults to vid; may be refined by downloaders).
              - resolution: Resolution label (e.g., "720p", "unknown").
              - video_fps: FPS rounded to an int.
              - ftp_download: True if a fresh FTP download/refresh occurred.

        Raises:
            RuntimeError: If the video cannot be found/downloaded or FPS is invalid.
        """
        ftp_download = False
        resolution = "unknown"
        video_title = vid

        # Default local path where we prefer to place downloaded assets.
        base_video_path = os.path.join(output_path, f"{vid}.mp4")

        # Respect Snellius policy: optionally disable YouTube fallback entirely.
        disable_yt = bool(getattr(config, "snellius_mode", False)) and bool(
            getattr(config, "snellius_disable_youtube_fallback", False)
        )

        if config.external_ssd:
            existing_path = os.path.join(output_path, f"{vid}.mp4")

            if os.path.exists(existing_path):
                # Use a temp directory so a failed refresh does not corrupt the SSD copy.
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
                    # If we successfully refreshed, validate it before replacing the SSD copy.
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

                    # Replace existing SSD copy with the refreshed one.
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

                # Refresh unavailable; keep existing file and validate metadata.
                logger.info(f"{vid}: FTP not available; using existing SSD copy.")
                self.set_video_title(video_title)
                video_fps = self.get_video_fps(existing_path)

                if self._fps_is_bad(video_fps):
                    # If SSD copy is corrupt and YouTube fallback is blocked, we must fail.
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

            # No SSD copy exists; download (FTP preferred) into output_path.
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

        # Search across provided cache locations to avoid redundant downloads.
        exists_somewhere = any(os.path.exists(os.path.join(path, f"{vid}.mp4")) for path in video_paths)

        if not exists_somewhere:
            # Not cached: attempt network retrieval (FTP first, then YouTube if allowed).
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

            # As a last local fallback, check the default output path.
            if os.path.exists(base_video_path):
                self.set_video_title(video_title)
                video_fps = self.get_video_fps(base_video_path)
                if self._fps_is_bad(video_fps):
                    raise RuntimeError(f"{vid}: invalid FPS on local fallback file.")
                logger.info(f"{vid}: found locally at {base_video_path}. fps={int(video_fps)}")  # type: ignore
                return base_video_path, video_title, resolution, int(video_fps), False  # type: ignore

            # No cached file and no successful download.
            raise RuntimeError(f"{vid}: video not found and download failed.")

        # Cached somewhere: select the first directory that contains the MP4.
        existing_folder = next((p for p in video_paths if os.path.exists(os.path.join(p, f"{vid}.mp4"))), None)
        use_folder = existing_folder if existing_folder else video_paths[-1]
        base_video_path = os.path.join(use_folder, f"{vid}.mp4")
        self.set_video_title(video_title)

        # Validate cached FPS to catch corrupt files early.
        video_fps = self.get_video_fps(base_video_path)
        if self._fps_is_bad(video_fps):
            raise RuntimeError(f"{vid}: invalid FPS on cached file.")

        logger.info(f"{vid}: using cached video at {base_video_path}. fps={int(video_fps)}")  # type: ignore
        return base_video_path, video_title, resolution, int(video_fps), False  # type: ignore

    def _wait_for_stable_file(self, src: str, checks: int = 2, interval: float = 0.5, timeout: float = 30) -> bool:
        """Wait until a file exists and its size stabilizes across consecutive checks.

        This is useful when the source file may still be in the process of being written,
        e.g., from another job or filesystem synchronization.

        Args:
            src: Path to the file being monitored.
            checks: Number of consecutive identical-size observations required.
            interval: Sleep time (seconds) between checks.
            timeout: Total time (seconds) to wait before returning False.

        Returns:
            True if the file became stable before timeout; otherwise False.
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
                # Track stability via repeated identical size measurements.
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
        """Copy a video to SSD using a temp file + atomic replace with retries.

        The copy process:
          1) Wait for the source file to become stable (size no longer changes).
          2) Copy to a temporary destination (<dest>.tmp).
          3) fsync (best-effort) then atomically replace temp -> final.
          4) Validate final size matches source size.
          5) Retry on I/O errors with linear backoff.

        Args:
            base_video_path: Source file path.
            internal_ssd: Destination directory (SSD).
            vid: Video identifier used to construct destination filename.
            max_attempts: Maximum copy attempts before raising.
            backoff: Sleep multiplier (seconds) between attempts.

        Returns:
            The final destination path on SSD.

        Raises:
            ValueError: If vid is empty.
            FileNotFoundError: If the source does not exist or never stabilizes.
            OSError/IOError: If copy fails after max_attempts or size validation fails.
        """
        # Defensive: ensure vid is usable for constructing filenames.
        if not vid or str(vid).strip() == "":
            raise ValueError("vid must be a non-empty string")

        dest_dir = internal_ssd
        dest = os.path.join(dest_dir, f"{vid}.mp4")
        tmp = dest + ".tmp"

        os.makedirs(dest_dir, exist_ok=True)

        # Short-circuit if base and dest are the same file (avoids redundant copy).
        try:
            if os.path.exists(dest) and os.path.exists(base_video_path) and os.path.samefile(base_video_path, dest):
                return dest
        except OSError:
            pass

        # Ensure the producer finished writing the file before we copy.
        if not self._wait_for_stable_file(base_video_path):
            raise FileNotFoundError(f"Source never became available or stayed unstable: {base_video_path}")

        # Capture expected size for post-copy validation.
        try:
            src_size = os.stat(base_video_path).st_size
        except OSError as e:
            raise FileNotFoundError(f"Cannot stat source: {base_video_path}: {e!r}")

        attempt = 0
        while True:
            attempt += 1
            try:
                # Copy metadata as well (timestamps/permissions) where possible.
                shutil.copy2(base_video_path, tmp)
                try:
                    # Best-effort fsync to reduce risk of silent partial writes.
                    with open(tmp, "rb") as f:
                        os.fsync(f.fileno())
                except Exception:
                    pass
                # Atomic replace: ensures consumers never see a half-written dest file.
                os.replace(tmp, dest)

                # Validate size to catch interrupted/partial copies.
                if os.stat(dest).st_size != src_size:
                    raise OSError(f"Size mismatch after copy: src={src_size}, dst={os.stat(dest).st_size}")

                return dest

            except (OSError, IOError):
                # Cleanup temp file to avoid poisoning subsequent attempts.
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
        """Download an MP4 from an HTTP file server that resembles an FTP listing.

        This implementation supports two approaches:
          1) Direct URL probing: /v/<alias>/files/<filename>.mp4
          2) Crawl fallback: traverse /v/<alias>/browse HTML pages to locate a link
             under /files/ matching the requested filename.

        HPC policy:
            If snellius_disable_downloads is enabled, this method logs an error and
            returns None without attempting network calls.

        Args:
            filename: Video id or filename stem; ".mp4" is appended if missing.
            base_url: Base URL of the file server (required).
            out_dir: Directory where the file will be saved.
            username: Optional basic-auth username.
            password: Optional basic-auth password.
            token: Optional query token passed as ?token=...
            timeout: HTTP request timeout (seconds).
            debug: Emit verbose debug logging when True.
            max_pages: Crawl safety limit to avoid unbounded traversal.

        Returns:
            A tuple (local_path, filename, resolution, fps) if found and downloaded,
            otherwise None.

        Notes:
            The second return element is the original filename stem (not necessarily
            the resolved/renamed output file path).
        """
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

        # The server appears to shard content under multiple aliases.
        aliases = ["tue1", "tue2", "tue3", "tue4"]
        req_params = {"token": token} if token else None

        logger.info(f"Starting download for '{filename_with_ext}'")
        if debug:
            logger.debug(
                f"Base URL: {base} | Auth: {'Basic' if username and password else 'None'} | Token: {'Yes' if token else 'No'}"  # noqa: E501
            )

        with requests.Session() as session:
            # Configure authentication and a stable User-Agent for server compatibility.
            if username and password:
                session.auth = (username, password)
            session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

            def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
                """GET a URL with optional streaming and standard error handling."""
                try:
                    r = session.get(url, timeout=timeout, params=req_params, stream=stream)
                    if debug:
                        logger.debug(f"GET {url} -> {r.status_code}")
                    if r.status_code == 401:
                        logger.error(f"Authentication failed for {url}")
                    r.raise_for_status()
                    return r
                except requests.RequestException as e:
                    # Network failures are expected; callers will try alternatives.
                    logger.warning(f"Request failed [{url}]: {e}")
                    return None

            # 1) Try direct /files paths (fastest when server layout is consistent).
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

                # Avoid overwriting an existing file by suffixing " (i)".
                if os.path.exists(local_path):
                    stem, suf = os.path.splitext(local_path)
                    i = 1
                    while os.path.exists(f"{stem} ({i}){suf}"):
                        i += 1
                    local_path = f"{stem} ({i}){suf}"
                    logger.warning(f"File exists, saving as: {local_path}")

                try:
                    # tqdm uses bytes; if content-length is missing, bar becomes indeterminate.
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

                # Attempt metadata extraction; failures are tolerated but logged.
                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path) or 0.0)
                    resolution = self.get_video_resolution_label(local_path)
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename, resolution, fps

            # 2) Crawl /browse fallback (HTML listing traversal).
            visited: Set[str] = set()

            def is_dir_link(href: str) -> bool:
                """Return True if an href points to a browse directory."""
                return href.startswith("/v/") and "/browse" in href

            def is_file_link(href: str) -> bool:
                """Return True if an href points to a downloadable file path."""
                return "/files/" in href

            def crawl(start_url: str) -> Optional[str]:
                """Depth-first crawl for a matching file link under the alias browse tree."""
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

                        # Match either the anchor text or final path segment.
                        if is_file_link(href):
                            anchor_text = (a.text or "").strip().lower()
                            tail = pathlib.PurePosixPath(urlparse(full).path).name.lower()
                            if anchor_text == filename_lower or tail == filename_lower:
                                logger.info(f"File located via crawl: {full}")
                                return full
                        if is_dir_link(href):
                            stack.append(full)

                return None

            # Try crawl on each alias until the file is found or aliases are exhausted.
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
        """Download a YouTube video using preferred resolutions with fallbacks.

        Primary path:
            - Use pytubefix to locate a stream that matches requested resolutions.

        Fallback path:
            - Use yt_dlp to pick a matching video format and convert to MP4 via ffmpeg.

        HPC policy:
            If snellius_disable_downloads is enabled, returns None without attempting
            network downloads.

        Args:
            vid: YouTube video ID.
            resolutions: Resolution preference order, e.g. ["720p", "480p", ...].
            output_path: Directory where the MP4 should be written.

        Returns:
            A tuple (video_file_path, vid, selected_resolution, fps) if successful,
            otherwise None.

        Notes:
            - pytubefix returns progressive/adaptive streams; the logic attempts to
              select <= 720p MP4 when an exact match isn't available.
            - yt_dlp branch uses tokens/cookies settings; Snellius mode avoids assuming
              a local browser is available for cookie extraction.
        """
        if self.snellius_mode and self.snellius_disable_downloads:
            logger.error(
                f"Snellius mode: downloads disabled (snellius_disable_downloads=True). "
                f"Please stage '{vid}.mp4' to /projects or your configured videos folder."
            )
            return None

        try:
            # Optional maintenance: upgrade packages weekly (Monday) if enabled.
            if self.update_package and datetime.datetime.today().weekday() == 0:
                maintenance_class.upgrade_package_if_needed("pytube")
                maintenance_class.upgrade_package_if_needed("pytubefix")

            youtube_url = f"https://www.youtube.com/watch?v={vid}"
            if self.need_authentication:
                # OAuth path, typically used when access restrictions apply.
                youtube_object = YouTube(youtube_url, self.client, use_oauth=True,  # type: ignore
                                         allow_oauth_cache=True, on_progress_callback=on_progress)
            else:
                youtube_object = YouTube(youtube_url, self.client, on_progress_callback=on_progress)  # type: ignore

            selected_stream = None
            selected_resolution = None

            # Attempt exact resolution matches in priority order.
            for resolution in resolutions:
                streams = youtube_object.streams.filter(res=resolution)
                if streams:
                    selected_resolution = resolution
                    selected_stream = streams.first() if hasattr(streams, "first") else streams[0]
                    break

            # fallback: pick best <= 720p mp4
            if not selected_stream:

                def _height_from_res(res_str: str) -> int:
                    """Parse an integer height from strings like '720p'; returns -1 if missing."""
                    if not res_str:
                        return -1
                    m = re.search(r"(\d{3,4})p", res_str)
                    return int(m.group(1)) if m else -1

                # Prefer progressive streams (include audio) if available; otherwise adaptive.
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

            # Store title for later reporting/metadata.
            self.video_title = youtube_object.title
            fps = self.get_video_fps(video_file_path)
            logger.info(f"{vid}: FPS={fps}.")
            return video_file_path, vid, str(selected_resolution), float(fps or 0.0)

        except Exception as e:
            # pytubefix can fail due to throttling, signature changes, or auth restrictions.
            logger.error(f"{vid}: pytubefix download failed: {e}")
            logger.info(f"{vid}: falling back to yt_dlp method.")

        # yt_dlp fallback
        try:
            # Optional maintenance: upgrade yt_dlp weekly (Monday) if enabled.
            if self.update_package and datetime.datetime.today().weekday() == 0:
                maintenance_class.upgrade_package_if_needed("yt_dlp")

            youtube_url = f"https://www.youtube.com/watch?v={vid}"

            # Extract metadata without downloading to enumerate formats.
            extract_opts = {"skip_download": True, "quiet": True}
            with yt_dlp.YoutubeDL(extract_opts) as ydl:  # type: ignore
                info_dict = ydl.extract_info(youtube_url, download=False)

            available_formats: list[dict[str, Any]] = info_dict.get("formats") or []
            selected_format_str = None
            selected_resolution = None

            # Prefer explicit height matches; try video-only first, then progressive.
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

            # Token required for some access patterns; sourced from internal secrets.
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
            # Final failure: caller will decide whether to error or continue.
            logger.error(f"{vid}: yt_dlp download failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Video info / trimming
    # -------------------------------------------------------------------------

    def get_video_fps(self, video_file_path: str) -> Optional[float]:
        """Retrieve the frames-per-second (FPS) of a video using OpenCV.

        Args:
            video_file_path: Path to the video file.

        Returns:
            FPS rounded to a whole number (as float), or None on failure.

        Notes:
            Some codecs/containers may cause CAP_PROP_FPS to return 0 or NaN; callers
            should validate using _fps_is_bad().
        """
        try:
            video = cv2.VideoCapture(video_file_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            video.release()
            return float(round(fps, 0))
        except Exception as e:
            logger.error(f"Failed to retrieve FPS: {e}")
            return None

    def get_video_resolution_label(self, video_path: str) -> str:
        """Return a resolution label derived from the video frame height.

        The method maps common heights (e.g., 720 -> "720p") and tolerates small
        deviations (e.g., 718 or 722) by snapping to the nearest standard height
        within a defined tolerance.

        Args:
            video_path: Path to a local video file.

        Returns:
            A string label such as "720p" or "<height>p" for non-standard heights.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If OpenCV cannot open the file.
        """
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

        # Snap near-standard heights to reduce noise from slightly-off encodes.
        tolerance = 8
        standard_heights = sorted(labels.keys())
        closest = min(standard_heights, key=lambda h: abs(h - height))
        if abs(closest - height) <= tolerance:
            return labels[closest]
        return f"{height}p"

    def trim_video(self, input_path: str, output_path: str, start_time: int, end_time: int) -> None:
        """Trim a video using MoviePy (re-encode) and write to output_path.

        This method is used as a compatibility fallback when ffmpeg stream-copy or
        progress parsing fails. It re-encodes using H.264 + AAC.

        Args:
            input_path: Source video file path.
            output_path: Destination file path.
            start_time: Start time (seconds).
            end_time: End time (seconds).

        Returns:
            None. Writes an MP4 file at output_path.
        """
        # MoviePy performs decoding/encoding; ensure destination directory exists.
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        video_clip.close()
