import os
import hashlib
from collections import defaultdict
import common

# ===== SETTINGS =====
DELETE_DUPLICATES = False  # <<< Set to True to delete smaller or identical duplicates
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


def file_hash(path, chunk_size=8192):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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

print("\n=== duplicate videos by filename ===\n")

for name, paths in sorted(duplicate_groups.items()):
    sized_paths = []
    for p in paths:
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        sized_paths.append((p, size))

    print(f"filename: {name}")
    for p, s in sized_paths:
        size_str = "unknown" if s < 0 else human_bytes(s)
        print(f"   {p}  ({size_str})")

    readable_sizes = [s for _, s in sized_paths if s >= 0]
    if not readable_sizes:
        print("   [skip] could not read any file sizes.\n")
        continue

    max_size = max(readable_sizes)
    largest_candidates = [p for p, s in sized_paths if s == max_size]
    keeper = largest_candidates[0]
    to_delete = [(p, s) for p, s in sized_paths if s >= 0 and s < max_size]

    # Handle equal-size duplicates
    equal_size_candidates = [p for p, s in sized_paths if s == max_size]
    if len(equal_size_candidates) > 1:
        print("   [note] equal-size duplicates detected, comparing content...")
        try:
            hashes = [(p, file_hash(p)) for p in equal_size_candidates]
            # Group by hash
            seen_hashes = {}
            for p, h in hashes:
                if h in seen_hashes:
                    # same hash, mark for deletion
                    s = os.path.getsize(p)
                    to_delete.append((p, s))
                else:
                    seen_hashes[h] = p
        except Exception as e:
            print(f"   [warn] hashing failed: {e}")

    if DELETE_DUPLICATES and to_delete:
        for p, s in to_delete:
            try:
                os.remove(p)
                deleted_files += 1
                bytes_freed += s
                print(f"   [deleted] {p} ({human_bytes(s)})")
            except OSError as e:
                print(f"   [error] failed to delete {p}: {e}")
        print(f"   [kept] {keeper} ({human_bytes(max_size)})\n")
    else:
        for p, s in to_delete:
            print(f"   [would delete] {p} ({human_bytes(s)})")
        print(f"   [would keep] {keeper} ({human_bytes(max_size)})\n")

print("=== summary ===")
print(f"duplicate groups: {total_groups}")
print(f"duplicate files (in those groups): {total_dupe_files}")
if DELETE_DUPLICATES:
    print(f"deleted files: {deleted_files}")
    print(f"bytes freed: {bytes_freed} ({human_bytes(bytes_freed)})")
else:
    potential = sum(max(0, len(paths) - 1) for paths in duplicate_groups.values())
    print(f"(dry run) potential deletions: {potential}")
    print("set DELETE_DUPLICATES = True at the top to remove smaller or identical duplicates.")
