#!/usr/bin/env python3
"""validate_release.py

Release validator for the dataset layout you actually have:

INPUTS (no parquet required)
---------------------------
1) mapping CSV (city/state rows) containing list-like columns:
   - videos, time_of_day, start_time, end_time, vehicle_type, upload_date, channel, continent, country, city, iso3, ...

2) mapping_metadata.csv (per-video metadata):
   - id, video, title, upload_date, channel, views, description, chapters, segments, date_updated

3) YOLO processed CSV folder:
   - one CSV per processed segment, filename pattern: {video_id}_{start_time}_{fps}.csv
     Example: XsQuWRGgGNk_5_60.csv

WHAT IT DOES
------------
- Builds paper-aligned rollups:
  * unique videos (uploads)
  * upload_records (sum of videos per mapping row)
  * segment_records (label-entry count = sum of time_of_day entries)
  * total duration (sum over (end-start) across segments)
  * continent rollups (segments + duration + shares)
  * continent day/night label-entry distribution
  * global + continent upload day/night composition (only_day / only_night / both)

- Validates internal consistency:
  * len(videos) matches len(time_of_day/start/end/vehicle_type) outer list per row
  * per-video segment alignment (start/end pairs)
  * start<end, non-negative

- Validates cross-file integrity:
  * mapping vs mapping_metadata (missing videos either side)
  * mapping_metadata.segments matches segments derived from mapping (deduped by (start,end))
  * YOLO CSV filenames map to known videos and known segment start times (with tolerance)
  * YOLO coverage: how many segments have a YOLO CSV

- Writes:
  * _output/validate_release_report.json
  * _output/validate_release.log
  * _output/sha256_manifest.txt

-----------------
Edit CONFIG below and run:
  python validate_release.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Allow very large CSV fields (mapping rows can contain extremely long list strings)
try:
    csv.field_size_limit(min(sys.maxsize, 1024 * 1024 * 1024))
except Exception:
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass


# =============================================================================
# CONFIG (edit these paths)
# =============================================================================
CONFIG: Dict[str, Any] = {
    "paths": {
        # Root folder that contains the mapping files and (optionally) _output/
        "data_dir": Path(__file__).resolve().parent,

        # REQUIRED
        "mapping_csv": "mapping.csv",
        "mapping_metadata_csv": "mapping_metadata.csv",

        # REQUIRED (folder of YOLO processed CSVs) - ABSOLUTE path is fine
        "yolo_dir": "/Volumes/Alam/pedestrians_in-youtube/data/bbox",

        # OPTIONAL: if provided, we will try to find claimed numbers in the LaTeX source
        "main_tex": None,  # e.g. "main.tex"
    },

    # Filename pattern for YOLO csvs
    "yolo_filename_regex": r"^(?P<video_id>[A-Za-z0-9_-]{6,})_(?P<start>[0-9]+(?:\.[0-9]+)?)_(?P<fps>[0-9]+(?:\.[0-9]+)?)\.csv$",  # noqa: E501

    # Allowed / expected categories
    "allowed": {
        "continents": ["Europe", "Asia", "North America", "Africa", "Oceania", "South America"],
        "time_of_day_codes": {0: "Day", 1: "Night"},
        "vehicle_type_codes": {
            0: "Car",
            1: "Bicycle",
            2: "Bus",
            3: "Truck",
            4: "Two-wheeler",
            5: "Monowheel/unicycle",
            6: "Electric scooter",
            7: "Automated car",
        },
    },

    # OPTIONAL: known expected totals from the paper/logs (set to None to skip)
    "expected": {
        "unique_videos": 32492,
        "duration_s": 57486804,
        "duration_h": 15968.56,
        "segment_records": 41595,
        "upload_records": 35132,
        "countries": 238,
        "rows_city_state_iso3": 5945,
    },

    # OPTIONAL: LaTeX number checks (works only if you set paths.main_tex)
    # Each item: (name, regex_to_capture_number, expected_key)
    # The regex MUST have exactly one capturing group for the numeric string.
    "latex_claim_checks": [
        (
            "unique_videos",
            r"(?i)\b(?:a\s+total\s+of|total\s+of)\s+([0-9]{1,3}(?:,[0-9]{3})*)\s+videos\b",
            "unique_videos",
        ),
        ("duration_hours", r"(?i)\b([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)\s+hours\b", "duration_h"),
        (
            "segment_records",
            r"(?i)\b([0-9]{1,3}(?:,[0-9]{3})*)\s+(?:segments|segment\s+records|label\s+entries)\b",
            "segment_records",
        ),
    ],
}

# Not in CONFIG on purpose (per your request)
YOLO_START_MATCH_TOLERANCE_S: float = 0.75


# =============================================================================
# Logging / report helpers
# =============================================================================
def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("validate_release")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-run in notebooks
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(out_dir / "validate_release.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _num_from_str(s: str) -> Optional[float]:
    if s is None:
        return None
    ss = str(s).strip().replace(",", "")
    if ss == "":
        return None
    try:
        return float(ss)
    except Exception:
        return None


def _int_from_any(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    try:
        return int(v)
    except Exception:
        try:
            f = float(str(v).strip())
            if math.isfinite(f):
                return int(f)
        except Exception:
            return None
    return None


def safe_eval_list(x: Any) -> List[Any]:
    """Parse list-like cell robustly."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []

    s = x.strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    # strip wrapping quotes if the entire cell is quoted
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in {"'", '"'}):
        s = s[1:-1].strip()

    if s == "" or s == "[]":
        return []

    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        # Fallback: [a,b,c] with bare tokens
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if inner == "":
                return []
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            return [p for p in parts if p != ""]
        return []


def to_list_of_ints(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[int] = []
        for t in x:
            iv = _int_from_any(t)
            if iv is not None:
                out.append(iv)
        return out
    iv = _int_from_any(x)
    return [iv] if iv is not None else []


def norm_list_of_lists_int(x: Any) -> List[List[int]]:
    lst = safe_eval_list(x)
    out: List[List[int]] = []
    for item in lst:
        if isinstance(item, list):
            arr = to_list_of_ints(item)
        else:
            arr = to_list_of_ints(item)
        out.append(arr if arr else [])
    return out


def norm_list_of_lists_float(x: Any) -> List[List[float]]:
    lst = safe_eval_list(x)
    out: List[List[float]] = []
    for item in lst:
        if isinstance(item, list):
            arr: List[float] = []
            for t in item:
                try:
                    fv = float(t)
                    if math.isfinite(fv):
                        arr.append(fv)
                except Exception:
                    continue
            out.append(arr)
        else:
            try:
                fv = float(item)
                if math.isfinite(fv):
                    out.append([fv])
                else:
                    out.append([])
            except Exception:
                out.append([])
    return out


def min_len(a: Sequence[Any], b: Sequence[Any]) -> int:
    return min(len(a), len(b))


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_float(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"


@dataclass
class Check:
    name: str
    ok: bool
    severity: str  # PASS/WARN/FAIL
    details: Dict[str, Any]


class Report:
    def __init__(self) -> None:
        self.checks: List[Check] = []

    def add(self, name: str, ok: bool, details: Optional[Dict[str, Any]] = None, warn: bool = False) -> None:
        if details is None:
            details = {}
        severity = "PASS" if ok else ("WARN" if warn else "FAIL")
        self.checks.append(Check(name=name, ok=ok, severity=severity, details=details))

    def summary(self) -> Dict[str, int]:
        c = Counter(ch.severity for ch in self.checks)
        return {"pass": int(c.get("PASS", 0)), "warn": int(c.get("WARN", 0)), "fail": int(c.get("FAIL", 0))}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "checks": [
                {"name": ch.name, "ok": ch.ok, "severity": ch.severity, "details": ch.details} for ch in self.checks
            ],
        }


# =============================================================================
# Core aggregation from mapping.csv
# =============================================================================
@dataclass
class VideoAgg:
    continents: Counter
    has_day: bool = False
    has_night: bool = False
    segments: Set[Tuple[float, float]] = None  # type: ignore

    def __post_init__(self):
        if self.segments is None:
            self.segments = set()


@dataclass
class MappingAgg:
    rows: int
    unique_city_state_iso3: int
    unique_countries: int
    upload_records: int
    unique_videos: int
    segment_records: int
    duration_s: float

    # Rollups
    continent_segment_records: Dict[str, int]
    continent_duration_s: Dict[str, float]
    continent_unique_uploads: Dict[str, int]

    continent_day_entries: Dict[str, int]
    continent_night_entries: Dict[str, int]

    global_upload_daynight: Dict[str, int]  # only_day, only_night, both_day_night, unknown
    continent_upload_daynight: Dict[str, Dict[str, int]]

    # Diagnostics
    n_rows_outer_mismatch: int
    n_videos_multi_continent: int


def parse_mapping_csv(path: Path, logger: logging.Logger, report: Report) -> Tuple[MappingAgg, Dict[str, VideoAgg]]:
    allowed_cont = set(CONFIG["allowed"]["continents"])
    cols: List[str] = []

    rows = 0
    city_state_iso3_keys: Set[str] = set()
    countries: Set[str] = set()

    upload_records = 0
    segment_records = 0
    duration_s = 0.0

    cont_seg: Dict[str, int] = defaultdict(int)
    cont_dur: Dict[str, float] = defaultdict(float)

    cont_day: Dict[str, int] = defaultdict(int)
    cont_night: Dict[str, int] = defaultdict(int)

    video_aggs: Dict[str, VideoAgg] = {}

    n_rows_outer_mismatch = 0
    n_bad_segment_pairs = 0
    bad_segment_examples: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []  # type: ignore
        required = {"continent", "country", "city", "iso3", "videos", "time_of_day", "start_time", "end_time"}
        missing = sorted(list(required - set(cols)))
        report.add(
            "mapping_csv.required_columns",
            ok=(len(missing) == 0),
            details={"missing": missing, "present": cols},
            warn=(len(missing) > 0),
        )

        for r in reader:
            rows += 1
            continent = (r.get("continent") or "").strip()
            country = (r.get("country") or "").strip()
            city = (r.get("city") or "").strip()
            iso3 = (r.get("iso3") or "").strip()
            state = (r.get("state") or "").strip()

            if country:
                countries.add(country)
            if city and iso3:
                city_state_iso3_keys.add(f"{city}|{state}|{iso3}")

            vids = [str(v).strip() for v in safe_eval_list(r.get("videos")) if str(v).strip() != ""]
            tod_lol = norm_list_of_lists_int(r.get("time_of_day"))
            st_lol = norm_list_of_lists_float(r.get("start_time"))
            en_lol = norm_list_of_lists_float(r.get("end_time"))
            veh_lol = norm_list_of_lists_int(r.get("vehicle_type")) if "vehicle_type" in set(cols) else []

            upload_records += len(vids)

            # Outer list alignment diagnostics: should match len(vids)
            outer_lens = {
                "videos": len(vids),
                "time_of_day": len(tod_lol),
                "start_time": len(st_lol),
                "end_time": len(en_lol),
            }
            if "vehicle_type" in set(cols):
                outer_lens["vehicle_type"] = len(veh_lol)

            if len({outer_lens[k] for k in outer_lens}) != 1:
                n_rows_outer_mismatch += 1

            # For rollups, trust only aligned portion (zip truncation)
            n_outer = len(vids)
            n_outer = min(n_outer, len(tod_lol), len(st_lol), len(en_lol))
            if "vehicle_type" in set(cols):
                n_outer = min(n_outer, len(veh_lol))

            # Per-row segment records defined as label-entry count (paper-aligned)
            row_seg_records = 0
            row_dur = 0.0

            if continent and continent not in allowed_cont:
                report.add("mapping_csv.continent_allowed", ok=False,
                           details={"continent": continent, "row": rows}, warn=True)

            for i in range(n_outer):
                vid = vids[i]
                if not vid:
                    continue

                tods = tod_lol[i] if i < len(tod_lol) else []
                starts = st_lol[i] if i < len(st_lol) else []
                ends = en_lol[i] if i < len(en_lol) else []

                # label-entry count for this video entry
                row_seg_records += len(tods)

                # duration: sum over aligned start/end; validate start/end pairs
                for s, e in zip(starts, ends):
                    try:
                        s2 = float(s)
                        e2 = float(e)
                    except Exception:
                        n_bad_segment_pairs += 1
                        if len(bad_segment_examples) < 25:
                            bad_segment_examples.append(
                                {"row": rows, "video": vid, "start": s, "end": e, "reason": "non_numeric"}
                            )
                        continue

                    if (not math.isfinite(s2)) or (not math.isfinite(e2)):
                        n_bad_segment_pairs += 1
                        if len(bad_segment_examples) < 25:
                            bad_segment_examples.append(
                                {"row": rows, "video": vid, "start": s2, "end": e2, "reason": "non_finite"}
                            )
                        continue

                    if s2 < 0 or e2 < 0 or e2 <= s2:
                        n_bad_segment_pairs += 1
                        if len(bad_segment_examples) < 25:
                            bad_segment_examples.append(
                                {"row": rows, "video": vid, "start": s2,
                                 "end": e2, "reason": "invalid_order_or_negative"}
                            )
                        continue

                    row_dur += (e2 - s2)

                # Per-video aggregation
                if vid not in video_aggs:
                    video_aggs[vid] = VideoAgg(continents=Counter())

                if continent:
                    video_aggs[vid].continents[continent] += 1

                # day/night flags
                for t in (tods or []):
                    if t == 0:
                        video_aggs[vid].has_day = True
                    elif t == 1:
                        video_aggs[vid].has_night = True

                # segments (dedup per video) only if valid start/end
                for s, e in zip(starts, ends):
                    try:
                        s2 = float(s)
                        e2 = float(e)
                        if math.isfinite(s2) and math.isfinite(e2) and (s2 >= 0) and (e2 > s2):
                            video_aggs[vid].segments.add((s2, e2))
                    except Exception:
                        continue

            # Continent label-entry distribution: count Day/Night entries
            day_ct = 0
            night_ct = 0
            for i in range(min_len(vids, tod_lol)):
                for t in tod_lol[i]:
                    if t == 0:
                        day_ct += 1
                    elif t == 1:
                        night_ct += 1

            if continent:
                cont_seg[continent] += row_seg_records
                cont_dur[continent] += row_dur
                cont_day[continent] += day_ct
                cont_night[continent] += night_ct

            segment_records += row_seg_records
            duration_s += row_dur

    # Canonical continent per video = mode
    video_canon_cont: Dict[str, str] = {}
    n_multi_cont = 0
    for vid, agg in video_aggs.items():
        if not agg.continents:
            continue
        if len(agg.continents) > 1:
            n_multi_cont += 1
        canon = sorted(agg.continents.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        video_canon_cont[vid] = canon

    # Unique uploads by continent (canonical)
    cont_unique: Dict[str, int] = defaultdict(int)
    for vid, cont in video_canon_cont.items():
        cont_unique[cont] += 1

    # Upload day/night composition global + by continent
    global_dn = Counter({"only_day": 0, "only_night": 0, "both_day_night": 0, "unknown": 0})
    cont_dn: Dict[str, Counter] = defaultdict(lambda: Counter({"only_day": 0,
                                                               "only_night": 0, "both_day_night": 0, "unknown": 0}))

    for vid, agg in video_aggs.items():
        if agg.has_day and agg.has_night:
            cat = "both_day_night"
        elif agg.has_day:
            cat = "only_day"
        elif agg.has_night:
            cat = "only_night"
        else:
            cat = "unknown"

        global_dn[cat] += 1
        cont = video_canon_cont.get(vid)
        if cont:
            cont_dn[cont][cat] += 1

    agg = MappingAgg(
        rows=rows,
        unique_city_state_iso3=len(city_state_iso3_keys),
        unique_countries=len(countries),
        upload_records=upload_records,
        unique_videos=len(video_aggs),
        segment_records=segment_records,
        duration_s=duration_s,
        continent_segment_records=dict(cont_seg),
        continent_duration_s=dict(cont_dur),
        continent_unique_uploads=dict(cont_unique),
        continent_day_entries=dict(cont_day),
        continent_night_entries=dict(cont_night),
        global_upload_daynight=dict(global_dn),
        continent_upload_daynight={k: dict(v) for k, v in cont_dn.items()},
        n_rows_outer_mismatch=n_rows_outer_mismatch,
        n_videos_multi_continent=n_multi_cont,
    )

    report.add(
        "mapping_csv.outer_list_alignment",
        ok=(n_rows_outer_mismatch == 0),
        details={"rows": rows, "rows_with_outer_len_mismatch": n_rows_outer_mismatch},
        warn=(n_rows_outer_mismatch > 0),
    )

    report.add(
        "mapping_csv.multi_continent_videos",
        ok=(n_multi_cont == 0),
        details={"videos_with_multiple_continents": n_multi_cont},
        warn=(n_multi_cont > 0),
    )

    report.add(
        "mapping_csv.segment_pairs_valid",
        ok=(n_bad_segment_pairs == 0),
        details={"bad_pair_count": n_bad_segment_pairs, "examples": bad_segment_examples},
        warn=(n_bad_segment_pairs > 0),
    )

    return agg, video_aggs


# =============================================================================
# mapping_metadata.csv checks
# =============================================================================
@dataclass
class MetadataRow:
    id: str
    video: str
    title: str
    upload_date: str
    channel: str
    views: str
    segments: str
    date_updated: str


def load_mapping_metadata(path: Path, logger: logging.Logger, report: Report) -> Dict[str, MetadataRow]:
    out: Dict[str, MetadataRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        required = {"video", "segments"}
        missing = sorted(list(required - set(cols)))
        report.add(
            "mapping_metadata.required_columns",
            ok=(len(missing) == 0),
            details={"missing": missing, "present": cols},
            warn=(len(missing) > 0),
        )

        for r in reader:
            video = (r.get("video") or "").strip()
            mr = MetadataRow(
                id=(r.get("id") or "").strip(),
                video=video,
                title=(r.get("title") or "").strip(),
                upload_date=(r.get("upload_date") or "").strip(),
                channel=(r.get("channel") or "").strip(),
                views=(r.get("views") or "").strip(),
                segments=(r.get("segments") or "").strip(),
                date_updated=(r.get("date_updated") or "").strip(),
            )
            if video:
                out[video] = mr

    report.add("mapping_metadata.unique_video_keys", ok=True, details={"n_rows": len(out)})
    return out


def compare_metadata_segments(video_aggs: Dict[str, VideoAgg], meta: Dict[str, MetadataRow], report: Report) -> None:
    mismatches: List[Dict[str, Any]] = []
    missing_meta: List[str] = []

    for vid, agg in video_aggs.items():
        mr = meta.get(vid)
        if mr is None:
            missing_meta.append(vid)
            continue

        derived = len(agg.segments)
        claimed = _int_from_any(mr.segments)

        if claimed is None:
            mismatches.append({"video": vid, "claimed": mr.segments, "derived": derived, "reason": "claimed_not_int"})
            continue

        if int(claimed) != int(derived):
            mismatches.append({"video": vid, "claimed": int(claimed), "derived": derived})

    report.add(
        "mapping_vs_metadata.missing_metadata_rows",
        ok=(len(missing_meta) == 0),
        details={"missing_count": len(missing_meta), "examples": missing_meta[:25]},
        warn=(len(missing_meta) > 0),
    )

    report.add(
        "mapping_vs_metadata.segment_count_match",
        ok=(len(mismatches) == 0),
        details={"mismatch_count": len(mismatches), "examples": mismatches[:25]},
        warn=(len(mismatches) > 0),
    )


# =============================================================================
# YOLO folder checks
# =============================================================================
@dataclass
class YoloFileInfo:
    path: str
    video_id: str
    start_time: float
    fps: float
    n_rows: int
    n_cols: int
    header: List[str]


def iter_csv_rowcount(path: Path, max_rows: Optional[int] = None) -> Tuple[int, int, List[str]]:
    """Count rows quickly without loading whole file; returns (rows, cols, header)."""
    n_rows = 0
    header: List[str] = []
    n_cols = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0, []
        n_cols = len(header)
        for _ in reader:
            n_rows += 1
            if max_rows is not None and n_rows >= max_rows:
                break
    return n_rows, n_cols, header


def check_yolo_folder(
    yolo_dir: Path,
    video_aggs: Dict[str, VideoAgg],
    report: Report,
    logger: logging.Logger,
) -> Tuple[List[YoloFileInfo], Dict[str, Any]]:
    rex = re.compile(CONFIG["yolo_filename_regex"])
    tol = float(YOLO_START_MATCH_TOLERANCE_S)

    # Build per-video set of known start times (from mapping)
    starts_by_video: Dict[str, Set[float]] = {}
    for vid, agg in video_aggs.items():
        starts_by_video[vid] = {s for (s, _e) in agg.segments}

    infos: List[YoloFileInfo] = []

    bad_name: List[str] = []
    unknown_video: List[str] = []
    unmatched_start: List[Dict[str, Any]] = []

    matched_segments: Set[Tuple[str, float]] = set()
    duplicate_match: List[Dict[str, Any]] = []

    if not yolo_dir.exists() or not yolo_dir.is_dir():
        report.add("yolo_dir.exists", ok=False, details={"path": str(yolo_dir)}, warn=False)
        return [], {"coverage": None}

    csv_files = sorted([p for p in yolo_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    report.add("yolo_dir.exists", ok=True, details={"path": str(yolo_dir), "n_csv": len(csv_files)})

    for p in csv_files:
        m = rex.match(p.name)
        if not m:
            bad_name.append(p.name)
            continue

        vid = m.group("video_id")
        st = float(m.group("start"))
        fps = float(m.group("fps"))

        if vid not in video_aggs:
            unknown_video.append(p.name)

        n_rows, n_cols, header = iter_csv_rowcount(p)
        infos.append(
            YoloFileInfo(
                path=str(p),
                video_id=vid,
                start_time=st,
                fps=fps,
                n_rows=n_rows,
                n_cols=n_cols,
                header=header[:50],
            )
        )

        if n_rows == 0:
            report.add("yolo_csv.non_empty", ok=False, details={"file": p.name}, warn=True)

        # Match start time to known segment start times
        known_starts = starts_by_video.get(vid, set())
        if known_starts:
            nearest = min(known_starts, key=lambda x: abs(x - st))
            if abs(nearest - st) <= tol:
                key = (vid, nearest)
                if key in matched_segments:
                    duplicate_match.append({"file": p.name, "video": vid, "start": st, "matched_start": nearest})
                matched_segments.add(key)
            else:
                unmatched_start.append(
                    {"file": p.name, "video": vid, "start": st,
                     "nearest_known_start": nearest, "delta": abs(nearest - st)}
                )
        else:
            unmatched_start.append({"file": p.name, "video": vid, "start": st, "reason": "no_known_segment_starts"})

    report.add(
        "yolo_filenames.pattern_match",
        ok=(len(bad_name) == 0),
        details={"bad_count": len(bad_name), "examples": bad_name[:50], "regex": CONFIG["yolo_filename_regex"]},
        warn=(len(bad_name) > 0),
    )

    report.add(
        "yolo_integrity.unknown_video_ids",
        ok=(len(unknown_video) == 0),
        details={"unknown_count": len(unknown_video), "examples": unknown_video[:50]},
        warn=(len(unknown_video) > 0),
    )

    report.add(
        "yolo_integrity.start_time_match",
        ok=(len(unmatched_start) == 0),
        details={"unmatched_count": len(unmatched_start), "examples": unmatched_start[:25], "tolerance_s": tol},
        warn=(len(unmatched_start) > 0),
    )

    report.add(
        "yolo_integrity.duplicate_segment_matches",
        ok=(len(duplicate_match) == 0),
        details={"duplicate_count": len(duplicate_match), "examples": duplicate_match[:25]},
        warn=(len(duplicate_match) > 0),
    )

    total_segments = sum(len(agg.segments) for agg in video_aggs.values())
    covered_segments = len(matched_segments)
    coverage = covered_segments / max(total_segments, 1)

    cov = {
        "total_segments": total_segments,
        "covered_segments": covered_segments,
        "coverage_pct": round(coverage * 100, 2),
    }
    report.add(
        "yolo_coverage.segment_coverage",
        ok=(coverage > 0.98) if total_segments > 0 else False,
        details=cov,
        warn=True,
    )

    return infos, cov


# =============================================================================
# Optional LaTeX checks
# =============================================================================
def _extract_numbers_from_latex(tex_path: Path, checks: List[Tuple[str, str, str]], report: Report) -> None:
    """Try to find paper-claimed numbers from main.tex using regex patterns."""
    text = tex_path.read_text(encoding="utf-8", errors="replace")

    for name, pattern, expected_key in checks:
        rex = re.compile(pattern)
        matches = rex.findall(text)
        if not matches:
            report.add(f"latex_claim:{name}", ok=False, details={"pattern": pattern, "reason": "no_match"}, warn=True)
            continue

        flat: List[str] = []
        for m in matches:
            if isinstance(m, tuple):
                flat.append(m[0])
            else:
                flat.append(m)

        if len(flat) > 1:
            report.add(
                f"latex_claim:{name}",
                ok=False,
                details={"pattern": pattern, "matches": flat[:25], "reason": "multiple_matches"},
                warn=True,
            )
            continue

        claimed_raw = flat[0]
        claimed_num = _num_from_str(claimed_raw)
        expected = CONFIG.get("expected", {}).get(expected_key)

        if claimed_num is None:
            report.add(f"latex_claim:{name}", ok=False, details={"claimed_raw": claimed_raw,
                                                                 "reason": "not_numeric"}, warn=True)
            continue

        if expected is None:
            report.add(f"latex_claim:{name}", ok=True, details={"claimed": claimed_num,
                                                                "expected": None, "note": "no_expected_config"})
            continue

        if isinstance(expected, int):
            ok = int(round(claimed_num)) == int(expected)
            report.add(f"latex_claim:{name}", ok=ok, details={"claimed": claimed_num,
                                                              "expected": expected}, warn=not ok)
        else:
            ok = abs(float(claimed_num) - float(expected)) <= 0.01
            report.add(
                f"latex_claim:{name}",
                ok=ok,
                details={"claimed": claimed_num, "expected": expected, "delta": float(claimed_num) - float(expected)},
                warn=not ok,
            )


# =============================================================================
# Pretty printing
# =============================================================================
def _print_mapping_summary(agg: MappingAgg, logger: logging.Logger) -> None:
    logger.info("\n=== Dataset summary (from mapping.csv) ===")
    logger.info(f"Rows (mapping): {fmt_int(agg.rows)}")
    logger.info(f"Unique city+state+iso3 keys: {fmt_int(agg.unique_city_state_iso3)}")
    logger.info(f"Unique countries/territories: {fmt_int(agg.unique_countries)}")
    logger.info(f"Upload records (sum of videos per row): {fmt_int(agg.upload_records)}")
    logger.info(f"Unique videos (global uploads): {fmt_int(agg.unique_videos)}")
    logger.info(f"Segment records (label entries): {fmt_int(agg.segment_records)}")
    logger.info(f"Duration: {fmt_int(int(round(agg.duration_s)))} s | {fmt_float(agg.duration_s/3600.0, 2)} h")


def _print_continent_table(agg: MappingAgg, logger: logging.Logger) -> None:
    order = CONFIG["allowed"]["continents"]
    seg_tot = sum(agg.continent_segment_records.get(c, 0) for c in order)
    dur_tot = sum(agg.continent_duration_s.get(c, 0.0) for c in order)

    logger.info("\n=== A) Continent segment distribution (label-entries; paper) ===")
    header = f"{'Continent':<14} {'Segments':>10} {'Seg%':>7} {'Dur(h)':>10} {'Dur%':>7}"
    logger.info(header)
    logger.info("-" * len(header))

    for c in order:
        seg = int(agg.continent_segment_records.get(c, 0))
        dur_h = float(agg.continent_duration_s.get(c, 0.0)) / 3600.0
        seg_pct = 100.0 * seg / max(seg_tot, 1)
        dur_pct = 100.0 * (dur_h * 3600.0) / max(dur_tot, 1e-9)
        logger.info(f"{c:<14} {seg:>10} {seg_pct:>6.2f} {dur_h:>10.2f} {dur_pct:>6.2f}")

    logger.info(f"{'Total':<14} {seg_tot:>10} {100.0:>6.2f} {dur_tot/3600.0:>10.2f} {100.0:>6.2f}")


def _print_daynight_entries(agg: MappingAgg, logger: logging.Logger) -> None:
    order = CONFIG["allowed"]["continents"]
    logger.info("\n=== D) Continent x time-of-day label entries (paper) ===")
    header = f"{'Continent':<14} {'Day':>12} {'Night':>12} {'Total':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    tot_day = 0
    tot_night = 0
    for c in order:
        d = int(agg.continent_day_entries.get(c, 0))
        n = int(agg.continent_night_entries.get(c, 0))
        t = d + n
        tot_day += d
        tot_night += n
        d_pct = (100.0 * d / max(t, 1))
        n_pct = (100.0 * n / max(t, 1))
        logger.info(
            f"{c:<14} {fmt_int(d):>12} ({d_pct:>6.2f}%) {fmt_int(n):>12} ({n_pct:>6.2f}%) {fmt_int(t):>12}"
        )

    tot = tot_day + tot_night
    logger.info(
        f"{'Total':<14} {fmt_int(tot_day):>12} ({100.0*tot_day/max(tot,1):>6.2f}%) "
        f"{fmt_int(tot_night):>12} ({100.0*tot_night/max(tot,1):>6.2f}%) {fmt_int(tot):>12}"
    )


def _print_upload_daynight(agg: MappingAgg, logger: logging.Logger) -> None:
    cats = ["only_day", "only_night", "both_day_night", "unknown"]
    tot = sum(int(agg.global_upload_daynight.get(k, 0)) for k in cats)
    logger.info("\n=== E) Upload day/night composition (global; paper) ===")
    for k in cats:
        v = int(agg.global_upload_daynight.get(k, 0))
        logger.info(f"{k:<16} {fmt_int(v):>10} ({100.0*v/max(tot,1):.2f}%)")
    logger.info(f"{'Total':<16} {fmt_int(tot):>10} (100.00%)")


# =============================================================================
# Main
# =============================================================================
def _resolve_path_maybe_absolute(base: Path, p: Any) -> Path:
    """If p is absolute, return it; else return base/p. Always .resolve()'d."""
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp.resolve()
    return (base / pp).resolve()


def _infer_yolo_root(yolo_dir: Path) -> Path:
    """
    Try to create a nicer 'root' for manifest paths.
    If yolo_dir looks like .../data/bbox, root becomes the parent of 'data'.
    """
    try:
        if yolo_dir.parent.name == "data":
            return yolo_dir.parent.parent
    except Exception:
        pass
    return yolo_dir


def _manifest_relpath(p_abs: Path, data_dir: Path, yolo_root: Path, yolo_dir: Path) -> str:
    """
    Avoid ValueError from Path.relative_to when files aren't under data_dir.
    Prefer relative_to(data_dir), then yolo_root, then yolo_dir, else filename.
    """
    try:
        return p_abs.relative_to(data_dir).as_posix()
    except ValueError:
        pass
    try:
        return p_abs.relative_to(yolo_root).as_posix()
    except ValueError:
        pass
    try:
        return p_abs.relative_to(yolo_dir).as_posix()
    except ValueError:
        return p_abs.name


def main() -> int:
    data_dir = Path(CONFIG["paths"]["data_dir"]).resolve()
    out_dir = data_dir / "_output"
    logger = _setup_logger(out_dir)
    report = Report()

    logger.info(f"Data dir: {data_dir}")

    # ---------------------------------------------------------------------
    # 0) Resolve required paths
    # ---------------------------------------------------------------------
    mapping_path = _resolve_path_maybe_absolute(data_dir, CONFIG["paths"]["mapping_csv"])
    meta_path = _resolve_path_maybe_absolute(data_dir, CONFIG["paths"]["mapping_metadata_csv"])
    yolo_dir = _resolve_path_maybe_absolute(data_dir, CONFIG["paths"]["yolo_dir"])
    yolo_root = _infer_yolo_root(yolo_dir)

    tex_rel = CONFIG["paths"].get("main_tex")
    tex_path = _resolve_path_maybe_absolute(data_dir, tex_rel) if tex_rel else None

    report.add("path.exists.mapping_csv", mapping_path.exists(), {"path": str(mapping_path)}, warn=False)
    report.add("path.exists.mapping_metadata_csv", meta_path.exists(), {"path": str(meta_path)}, warn=False)
    report.add("path.exists.yolo_dir", yolo_dir.exists(), {"path": str(yolo_dir)}, warn=False)
    if tex_path is not None:
        report.add("path.exists.main_tex", tex_path.exists(), {"path": str(tex_path)}, warn=True)

    if not mapping_path.exists() or not meta_path.exists() or not yolo_dir.exists():
        logger.error("Missing required inputs. Edit CONFIG['paths'].")
        (out_dir / "validate_release_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return 1

    # ---------------------------------------------------------------------
    # 1) SHA256 manifest
    # ---------------------------------------------------------------------
    sha_lines: List[str] = []

    for p in [mapping_path, meta_path]:
        d = sha256_file(p)
        sha_lines.append(f"{d}  {p.name}")

    if yolo_dir.is_dir():
        for p in sorted([x for x in yolo_dir.rglob("*") if x.is_file()]):
            if p.suffix.lower() not in {".csv", ".txt", ".json", ".md"}:
                continue
            d = sha256_file(p)
            rel = _manifest_relpath(p.resolve(), data_dir=data_dir, yolo_root=yolo_root, yolo_dir=yolo_dir)
            sha_lines.append(f"{d}  {rel}")

    (out_dir / "sha256_manifest.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    report.add("sha256_manifest_written", True,
               {"path": str(out_dir / "sha256_manifest.txt"), "n_entries": len(sha_lines)})

    # ---------------------------------------------------------------------
    # 2) Parse mapping.csv and compute paper-aligned rollups
    # ---------------------------------------------------------------------
    mapping_agg, video_aggs = parse_mapping_csv(mapping_path, logger, report)

    _print_mapping_summary(mapping_agg, logger)
    _print_continent_table(mapping_agg, logger)
    _print_daynight_entries(mapping_agg, logger)
    _print_upload_daynight(mapping_agg, logger)

    # ---------------------------------------------------------------------
    # 3) Compare to expected totals (if provided)
    # ---------------------------------------------------------------------
    exp = CONFIG.get("expected", {})

    def _cmp(name: str, computed: Any, expected_key: str, tol: float = 0.0) -> None:
        expected = exp.get(expected_key)
        if expected is None:
            report.add(f"expected:{name}", True, {"computed": computed, "expected": None, "note": "not_configured"})
            return
        if isinstance(expected, int):
            ok = int(computed) == int(expected)
            report.add(f"expected:{name}", ok, {"computed": int(computed), "expected": int(expected)}, warn=not ok)
        else:
            ok = abs(float(computed) - float(expected)) <= tol
            report.add(
                f"expected:{name}",
                ok,
                {"computed": float(computed), "expected": float(expected), "delta": float(computed) - float(expected)},
                warn=not ok,
            )

    _cmp("unique_videos", mapping_agg.unique_videos, "unique_videos")
    _cmp("upload_records", mapping_agg.upload_records, "upload_records")
    _cmp("segment_records", mapping_agg.segment_records, "segment_records")
    _cmp("duration_s", int(round(mapping_agg.duration_s)), "duration_s")
    _cmp("duration_h", round(mapping_agg.duration_s / 3600.0, 2), "duration_h", tol=0.01)
    _cmp("countries", mapping_agg.unique_countries, "countries")
    _cmp("rows_city_state_iso3", mapping_agg.unique_city_state_iso3, "rows_city_state_iso3")

    # ---------------------------------------------------------------------
    # 4) Load mapping_metadata.csv and cross-check
    # ---------------------------------------------------------------------
    meta = load_mapping_metadata(meta_path, logger, report)

    compare_metadata_segments(video_aggs, meta, report)

    missing_in_mapping = [vid for vid in meta.keys() if vid not in video_aggs and vid != "#NAME?"]
    report.add(
        "metadata_vs_mapping.missing_in_mapping",
        ok=(len(missing_in_mapping) == 0),
        details={"missing_count": len(missing_in_mapping), "examples": missing_in_mapping[:25]},
        warn=(len(missing_in_mapping) > 0),
    )

    # ---------------------------------------------------------------------
    # 5) Check YOLO folder
    # ---------------------------------------------------------------------
    _infos, _cov = check_yolo_folder(yolo_dir, video_aggs, report, logger)

    # ---------------------------------------------------------------------
    # 6) Optional: validate LaTeX claimed numbers
    # ---------------------------------------------------------------------
    if tex_path is not None and tex_path.exists():
        _extract_numbers_from_latex(tex_path, CONFIG.get("latex_claim_checks", []), report)

    # ---------------------------------------------------------------------
    # 7) Write report
    # ---------------------------------------------------------------------
    report_path = out_dir / "validate_release_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    summ = report.summary()
    logger.info("\n=== Validation summary ===")
    logger.info(json.dumps(summ, indent=2))
    logger.info(f"Report: {report_path}")

    return 1 if summ.get("fail", 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
