import os

def find_common_mp4(folder1, folder2):
    # Get list of mp4 files in each folder (ignore case)
    files1 = {f.lower() for f in os.listdir(folder1) if f.lower().endswith('.mp4')}
    files2 = {f.lower() for f in os.listdir(folder2) if f.lower().endswith('.mp4')}

    # Find intersection
    common_files = files1.intersection(files2)

    if common_files:
        print("Common .mp4 files in both folders:")
        for filename in sorted(common_files):
            print(filename)
    else:
        print("No common .mp4 files found.")

# Example usage:
# Replace 'path_to_folder1' and 'path_to_folder2' with your actual folder paths
find_common_mp4('/media/salam/TUeMobility/pedestrians-in-youtube/videos', '/media/salam/Mobility/pedestrians-in-youtube/videos')
