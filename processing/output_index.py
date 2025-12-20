"""Output indexing utilities.

This module provides fast discovery of existing output artifacts so the pipeline
can skip already-processed segments without expensive per-segment filesystem
globbing.

The indexing intentionally ignores FPS in filenames. This matches the pipeline's
definition of "DONE": if any CSV exists for a (video_id, start_time) pair,
the segment is treated as completed for that mode.
"""

import os
from typing import Dict, List, Set, Tuple


class OutputIndex:
    """Indexes existing bbox/seg CSV outputs for fast 'already done' checks.

    Expected filename convention:
      {video_id}_{start_time}_{fps}.csv

    Important:
      - video_id may contain underscores, so parsing must split from the right.
      - fps is intentionally ignored in the "DONE" decision.
    """

    def __init__(self) -> None:
        """Initializes the output indexer.

        This class is stateless. Indexing is intended to run once per pass over
        the input mapping so filesystem scanning is amortized across segments.
        """
        # No instance state required.
        # Keeping the constructor makes it easy to extend later (e.g., caching).
        pass

    def _index_existing_outputs(self, data_folders: List[str], want_bbox: bool,
                                want_seg: bool) -> Dict[str, Set[Tuple[str, int]]]:
        """Indexes existing outputs once per pass.

        Why this exists:
          Repeated globbing for each segment is expensive, especially when
          `mapping.csv` is large. This function scans output folders once and
          stores results in sets for O(1) membership checks.

        "DONE" definition:
          We index by (video_id, start_time) and intentionally ignore FPS.
          If any file matches:
            data/*/bbox/{video_id}_{start}_*.csv
          then that (video_id, start) is treated as done for bbox mode
          (and similarly for seg mode).

        Args:
          data_folders: List of base output folders (e.g., ["data1", "data2"]).
            Each folder may contain "bbox/" and/or "seg/" subfolders.
          want_bbox: Whether to scan bbox outputs.
          want_seg: Whether to scan segmentation outputs.

        Returns:
          Dictionary with two sets:
            {
              "bbox_start": set([(video_id, start_int), ...]),
              "seg_start":  set([(video_id, start_int), ...]),
            }
        """
        # Use sets for fast membership checks and to deduplicate entries.
        # Tuple values are hashable and compact for indexing.
        bbox_start: Set[Tuple[str, int]] = set()
        seg_start: Set[Tuple[str, int]] = set()

        def _scan_dir(base_folder: str, sub: str, out_set: Set[Tuple[str, int]]) -> None:
            """Scans one subdirectory ("bbox" or "seg") and populates `out_set`.

            Parsing is intentionally tolerant:
              - Non-CSV files are ignored.
              - Unexpected filenames are ignored (no crash).
              - Underscores in video_id are supported via rsplit.

            Args:
              base_folder: Base output folder, already absolute.
              sub: Subfolder to scan ("bbox" or "seg").
              out_set: Set to populate with (video_id, start_time_int).

            Returns:
              None.
            """
            # Build the folder path we want to inspect.
            d = os.path.join(base_folder, sub)

            # If it doesn't exist, there is nothing to do.
            if not os.path.isdir(d):
                return

            # scandir is typically faster than listdir + stat calls.
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        # We only index files (skip dirs/symlinks).
                        if not entry.is_file():
                            continue

                        # Only accept final CSVs (partials belong elsewhere).
                        name = entry.name
                        if not name.endswith(".csv"):
                            continue

                        # Drop ".csv" extension to parse the stem.
                        stem = name[:-4]

                        # Expected pattern: "{vid}_{start}_{fps}"
                        # We ignore fps; we only need (vid, start).
                        try:
                            # rsplit from the right: preserves underscores in video_id.
                            vid_part, st_str, _fps_str = stem.rsplit("_", 2)
                            start_time = int(st_str)
                        except Exception:
                            # Filename doesn't match the expected convention.
                            # Ignore rather than failing the full scan.
                            continue

                        # Record presence of any output for this (video_id, start_time).
                        # This is the key logic that makes "DONE ignores FPS" work.
                        out_set.add((vid_part, start_time))

            except FileNotFoundError:
                # Directory may disappear between isdir() and scandir() due to races.
                # Treat as empty and continue.
                return

        # Scan all configured data folders.
        # Normalize to absolute paths so a folder referenced in multiple ways
        # does not cause redundant scanning or inconsistent results.
        for folder in data_folders:
            base = os.path.abspath(folder)

            # Scan each mode only if requested.
            # This avoids unnecessary filesystem work.
            if want_bbox:
                _scan_dir(base, "bbox", bbox_start)
            if want_seg:
                _scan_dir(base, "seg", seg_start)

        # Return a stable structure with two named sets.
        return {"bbox_start": bbox_start, "seg_start": seg_start}
