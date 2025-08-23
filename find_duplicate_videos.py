# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
from collections import defaultdict
import common

# your folders
folders = common.get_configs('videos')

# collect filenames
name_map = defaultdict(list)

for folder in folders:
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
                path = os.path.join(root, f)
                name_map[f].append(path)

# report duplicates
print("\n=== duplicate videos by filename ===\n")
for name, paths in name_map.items():
    if len(paths) > 1:
        print(f"filename: {name}")
        for p in paths:
            print("  ", p)
        print()
