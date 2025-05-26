# by Shadab Alam <md_shadab_alam@outlook.com>
import numpy as np
import common
from custom_logger import CustomLogger
from logmod import logs
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class Geometry():
    def __init__(self) -> None:
        pass

    def reassign_ids_directional_cross_fix(self, df, distance_threshold=0.05, yolo_ids=None):
        """
        Reassign object IDs based on consistent movement direction along the X-axis.

        Handles ID switching problems when objects cross paths, especially useful for normalized
        YOLO detection/tracking outputs. Tracks can be filtered by specific YOLO classes.


        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame containing YOLO tracking data.
            Expected columns: ['YOLO_id', 'X-center', 'Y-center', 'Width', 'Height', 'Unique Id', 'Frame Count']

        distance_threshold : float, optional
            Max normalized X distance to associate detections with tracks. Default is 0.05 (i.e., 5% of frame width).

        yolo_ids : list of int or None, optional
            List of YOLO class IDs (e.g., [0] for 'person') to filter tracking on.
            If None, all YOLO IDs are included.

        Returns
        -------
        pandas.DataFrame
            DataFrame with added 'New Id' column for corrected object tracking IDs.
        """

        # Load the CSV and sort it by frame number to process in time order
        df = df.sort_values(by='Frame Count')

        # If filtering by specific YOLO classes, keep only those rows
        if yolo_ids is not None:
            df = df[df['YOLO_id'].isin(yolo_ids)]

        # Create a new column to store the updated, corrected tracking ID
        df['New Id'] = -1

        # Initialize ID counter for assigning new identities
        next_id = 0

        # Dictionary to store active object tracks
        # Format: {track_id: {'x': last_x_position, 'dir': motion_direction}}
        active_tracks = {}

        # Iterate through each frame
        for frame in sorted(df['Frame Count'].unique()):
            # Get detections for this frame
            frame_detections = df[df['Frame Count'] == frame].copy()
            # X positions for all detections in this frame
            detections = frame_detections[['X-center']].to_numpy().flatten()

            # Track used detections and track IDs
            assigned_detections = set()
            used_track_ids = set()

            # Match each detection to a known track
            for det_idx, x in enumerate(detections):
                best_match_id = None
                best_dist = distance_threshold + 1

                for track_id, track in active_tracks.items():
                    if track_id in used_track_ids:
                        continue  # Skip if track already used in this frame

                    prev_x = track['x']
                    direction = track['dir']
                    dx = x - prev_x
                    dist = abs(dx)

                    # Only consider matches that preserve direction
                    if dist < best_dist:
                        new_direction = np.sign(dx)
                        if direction is None or new_direction == direction:
                            best_match_id = track_id
                            best_dist = dist

                # Get original DataFrame index for this detection
                det_index = frame_detections.index[det_idx]

                if best_match_id is not None:
                    # Assign detection to matched track
                    df.at[det_index, 'New Id'] = best_match_id
                    dx = x - active_tracks[best_match_id]['x']
                    new_direction = np.sign(dx) if dx != 0 else active_tracks[best_match_id]['dir']
                    active_tracks[best_match_id] = {'x': x, 'dir': new_direction}
                    used_track_ids.add(best_match_id)
                    assigned_detections.add(det_idx)
                else:
                    # No match found → create new track
                    df.at[det_index, 'New Id'] = next_id
                    active_tracks[next_id] = {'x': x, 'dir': None}
                    next_id += 1

        return df
