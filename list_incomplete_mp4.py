#!/usr/bin/env python3
"""
list_bad_videos.py

Print ONLY files that appear incomplete/corrupted — including those that play
but cut off early. Combines:
  1) Metadata checks (ffprobe): duration, has A/V streams
  2) Fallback structure check: 'moov' atom (if no ffprobe)
  3) Duration outlier vs folder median (ffprobe)
  4) Full decode scan (ffmpeg) to catch mid-file truncation

OUTPUT: one bad file path per line.
"""

import sys
import os
import json
import shutil
import subprocess
import statistics
from pathlib import Path
import common

# ---------------- SETTINGS (edit as you like) ----------------
DIRECTORIES = common.get_configs("videos")          # used if no CLI paths are passed
EXTENSIONS = {".mp4", ".m4v", ".mov"}
MIN_SIZE_BYTES = 150 * 1024         # tiny files => bad

# ffprobe-based metadata checks
MIN_DURATION_SECONDS = 5.0          # duration <= this => bad
REQUIRE_STREAMS = True              # require at least one audio or video stream

# Outlier detection (needs ffprobe)
DURATION_OUTLIER_RATIO = 0.92       # flag if duration < 92% of folder median
MIN_MISSING_SECONDS = 8 * 60        # and missing at least this much time

# Decode scan (needs ffmpeg)
RUN_DECODE_CHECK = True             # set False to skip decode pass (faster)
DECODE_TIMEOUT_PER_GB = 1200        # seconds per GB (upper bound timeout)

# Output style
PRINT_REASONS = False               # if True, print "path || reason"
# --------------------------------------------------------------


def has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ffprobe_full(path: Path):
    """
    Return (duration_seconds, has_av_stream, error_msg)
    """
    try:
        p = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path)
            ],
            capture_output=True, text=True
        )
        if p.returncode != 0:
            return None, False, p.stderr.strip() or "ffprobe non-zero exit"
        data = json.loads(p.stdout or "{}")

        dur = None
        if "format" in data and "duration" in data["format"]:
            try:
                dur = float(data["format"]["duration"])
            except (TypeError, ValueError):
                dur = None

        streams = data.get("streams", [])
        has_av = any(s.get("codec_type") in ("video", "audio") for s in streams)
        return dur, has_av, None
    except Exception as e:
        return None, False, f"ffprobe exception: {e}"


def quick_mp4_has_moov(path: Path, scan_bytes: int = 2_000_000) -> bool:
    """
    Light structural hint: look for 'moov' atom near head/tail.
    """
    try:
        size = path.stat().st_size
        if size == 0:
            return False
        with open(path, "rb") as f:
            head = f.read(min(size, scan_bytes))
            if b"moov" in head:
                return True
            if size > scan_bytes:
                f.seek(-scan_bytes, os.SEEK_END)
                tail = f.read(scan_bytes)
                if b"moov" in tail:
                    return True
        return False
    except Exception:
        return False


FFMPEG_BAD_SIGNS = (
    "error", "invalid", "corrupt", "truncated", "malformed",
    "end of file", "unexpected eof", "packet corrupt", "overread",
    "concealing", "missing picture", "missing keyframe"
)

def ffmpeg_decode_ok(path: Path) -> bool:
    """
    Fully decodes the file to the null muxer. Returns True if ffmpeg reports no errors.
    """
    size = path.stat().st_size
    timeout = max(300, int((size / (1024**3)) * DECODE_TIMEOUT_PER_GB) + 60)

    try:
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-xerror",       # escalate certain issues to errors
            "-nostdin",
            "-i", str(path),
            "-f", "null",
            "-"              # write to null muxer
        ]
        p = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, timeout=timeout)
        stderr = (p.stderr or "").lower()
        if p.returncode != 0:
            return False
        # Even with 0 exit, scan for suspicious messages
        return not any(sig in stderr for sig in FFMPEG_BAD_SIGNS)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def iter_videos(paths):
    for root in paths:
        p = Path(root)
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            yield p
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in EXTENSIONS:
                    yield f


def main():
    ffprobe_ok = has("ffprobe")
    ffmpeg_ok = has("ffmpeg")

    folders = sys.argv[1:] if len(sys.argv) > 1 else DIRECTORIES
    files = list(iter_videos(folders))
    if not files:
        return 0

    # 1) First pass: metadata/structural checks + collect durations for outlier step
    durations = {}
    reasons = {}  # path -> list of reasons
    bad = set()

    for f in files:
        r = []
        try:
            size = f.stat().st_size
        except Exception:
            bad.add(f); reasons.setdefault(f, []).append("unreadable"); continue

        if size < MIN_SIZE_BYTES:
            bad.add(f); reasons.setdefault(f, []).append(f"tiny<{MIN_SIZE_BYTES}B")
            durations[f] = 0.0
            continue

        if ffprobe_ok:
            dur, has_av, err = ffprobe_full(f)
            if err:
                bad.add(f); reasons.setdefault(f, []).append(f"ffprobe:{err}")
                durations[f] = 0.0
                continue
            durations[f] = float(dur) if dur else 0.0

            if REQUIRE_STREAMS and not has_av:
                bad.add(f); reasons.setdefault(f, []).append("no audio/video streams")

            if dur is None or dur <= 0:
                bad.add(f); reasons.setdefault(f, []).append("duration<=0")
            elif dur < MIN_DURATION_SECONDS:
                bad.add(f); reasons.setdefault(f, []).append(f"short<{MIN_DURATION_SECONDS}s")
        else:
            # no ffprobe: do structural 'moov' check
            durations[f] = 0.0  # we can't compute outliers later
            if not quick_mp4_has_moov(f):
                bad.add(f); reasons.setdefault(f, []).append("missing moov")

    # 2) Outlier detection per folder (only if we have usable durations)
    if ffprobe_ok:
        per_folder = {}
        for f, d in durations.items():
            per_folder.setdefault(f.parent, []).append(d)

        folder_median = {}
        for folder, durs in per_folder.items():
            vals = [d for d in durs if d > 0]
            folder_median[folder] = statistics.median(vals) if vals else 0.0

        for f, dur in durations.items():
            if f in bad:
                continue
            med = folder_median.get(f.parent, 0.0)
            if med >= (MIN_MISSING_SECONDS + 60):  # only if siblings are long enough
                if dur < DURATION_OUTLIER_RATIO * med and (med - dur) >= MIN_MISSING_SECONDS:
                    bad.add(f); reasons.setdefault(f, []).append(
                        f"outlier:{dur:.0f}s<<median~{med:.0f}s"
                    )

    # 3) Decode pass to catch mid-stream truncations (optional)
    if RUN_DECODE_CHECK and ffmpeg_ok:
        for f in files:
            if f in bad:
                continue
            try:
                if not ffmpeg_decode_ok(f):
                    bad.add(f); reasons.setdefault(f, []).append("decode_errors")
            except Exception:
                bad.add(f); reasons.setdefault(f, []).append("decode_exception")

    # 4) Print results: ONLY bad files (one per line). Optionally with reasons.
    for f in sorted(bad, key=lambda x: str(x).lower()):
        if PRINT_REASONS:
            print(f"{f} || {', '.join(reasons.get(f, []))}")
        else:
            print(f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
