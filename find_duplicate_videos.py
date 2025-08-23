import os
import hashlib
from collections import defaultdict
import common

# your folders
folders = common.get_configs('videos')


def file_hash(path, algo="md5", chunk_size=8192):
    """compute hash of a file"""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# collect hashes
hash_map = defaultdict(list)

for folder in folders:
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
                path = os.path.join(root, f)
                try:
                    h = file_hash(path)
                    hash_map[h].append(path)
                except Exception as e:
                    print(f"error with {path}: {e}")

# report duplicates
print("\n=== duplicate videos found ===\n")
for h, paths in hash_map.items():
    if len(paths) > 1:
        print(f"hash: {h}")
        for p in paths:
            print("  ", p)
        print()
