# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
from collections import defaultdict
import common
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger

# ===== SETTINGS =====
DELETE_DUPLICATES = False  # <<< Set to True to delete smaller duplicates
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")
# =====================


def human_bytes(n: int) -> str:
    """Convert bytes to human-readable units."""
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


# your folders
folders = common.get_configs('videos')

# collect filenames
name_map = defaultdict(list)
for folder in folders:
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                path = os.path.join(root, f)
                name_map[f].append(path)

# find duplicates
duplicate_groups = {name: paths for name, paths in name_map.items() if len(paths) > 1}

total_groups = len(duplicate_groups)
total_dupe_files = sum(len(v) for v in duplicate_groups.values())
deleted_files = 0
bytes_freed = 0

logger.info("\n=== duplicate videos by filename ===\n")

for name, paths in sorted(duplicate_groups.items()):
    sized_paths = []
    for p in paths:
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        sized_paths.append((p, size))

    logger.info(f"filename: {name}")
    for p, s in sized_paths:
        size_str = "unknown" if s < 0 else human_bytes(s)
        logger.info(f"   {p}  ({size_str})")

    readable_sizes = [s for _, s in sized_paths if s >= 0]
    if not readable_sizes:
        logger.info("   [skip] could not read any file sizes.\n")
        continue

    max_size = max(readable_sizes)
    largest_candidates = [p for p, s in sized_paths if s == max_size]
    keeper = largest_candidates[0]
    to_delete = [(p, s) for p, s in sized_paths if s >= 0 and s < max_size]

    if not to_delete and len(largest_candidates) > 1:
        logger.info("   [note] multiple files have the same (largest) size; none deleted automatically.\n")
        continue

    if DELETE_DUPLICATES and to_delete:
        for p, s in to_delete:
            try:
                os.remove(p)
                deleted_files += 1
                bytes_freed += s
                logger.info(f"   [deleted] {p} ({human_bytes(s)})")
            except OSError as e:
                logger.info(f"   [error] failed to delete {p}: {e}")
        logger.info(f"   [kept] {keeper} ({human_bytes(max_size)})\n")
    else:
        for p, s in to_delete:
            logger.info(f"   [would delete] {p} ({human_bytes(s)})")
        logger.info(f"   [would keep] {keeper} ({human_bytes(max_size)})\n")

logger.info("=== summary ===")
logger.info(f"duplicate groups: {total_groups}")
logger.info(f"duplicate files (in those groups): {total_dupe_files}")
if DELETE_DUPLICATES:
    logger.info(f"deleted files: {deleted_files}")
    logger.info(f"bytes freed: {bytes_freed} ({human_bytes(bytes_freed)})")
else:
    potential = sum(max(0, len(paths) - 1) for paths in duplicate_groups.values())
    logger.info(f"(dry run) potential deletions: {potential}")
    logger.info("set DELETE_DUPLICATES = True at the top to remove smaller duplicates.")
