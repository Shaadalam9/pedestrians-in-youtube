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
        Reassigns object tracking IDs to correct for identity switches when objects cross paths,
        using consistent X-axis movement and optional class filtering.

        This function is designed for post-processing YOLO (or similar) tracking outputs in video data,
        where objects may switch tracking IDs after crossing each other. The algorithm matches
        detections across frames based on spatial proximity along the X-axis and maintains consistent
        movement direction to reduce erroneous ID swaps. Each detection is assigned a new 'New Id'
        that attempts to reflect the real-world object more consistently over time.

        Main Features:
        --------------
        - Iterates through frames in temporal order and attempts to associate detections with existing tracks.
        - Tracks are maintained by their most recent X position and direction of movement.
        - A detection is matched to an existing track only if it is within a specified maximum normalized X distance
            ("distance_threshold") and moving in the same direction as the track's historical movement.
        - If no suitable track is found, the detection starts a new track (gets a new ID).
        - Optionally, tracks can be restricted to specific YOLO class IDs (e.g., only 'person' objects).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of detection and tracking data, with the following required columns:
                - 'YOLO_id'    : Integer class ID from YOLO detector.
                - 'X-center'   : Normalized X coordinate of bounding box center (0.0 to 1.0).
                - 'Y-center'   : Normalized Y coordinate of bounding box center (not used by this function).
                - 'Width'      : Bounding box width (not used here).
                - 'Height'     : Bounding box height (not used here).
                - 'Unique Id'  : Original tracking ID (not used in new ID assignment).
                - 'Frame Count': Frame number (must be sortable in time order).

        distance_threshold : float, optional
            Maximum allowed normalized X distance (0 to 1) to associate a detection with an existing track.
            Detections farther apart than this threshold are not considered part of the same object.
            The default (0.05) means detections within 5% of the frame width can be matched.

            **Note:** Setting this threshold too low can cause excessive fragmentation of tracks (new IDs
            created too often). Setting it too high can cause unrelated objects to be merged under the same ID.

        yolo_ids : list of int or None, optional
            List of YOLO class IDs to include in processing. Only detections with 'YOLO_id' in this list
            will be tracked and reassigned. If None, all detected object classes are included.

        Returns
        -------
        pandas.DataFrame
            Copy of the input DataFrame with an added 'New Id' column. This column contains the
            corrected object tracking IDs, designed to minimize identity switches due to objects
            crossing paths. All other columns from the original DataFrame are preserved.

        Limitations & Notes
        -------------------
        - This method assumes objects move mainly along the X-axis, and swaps are most common there.
            It is not suitable for arbitrary or erratic movement patterns, or for scenes where Y-axis
            motion is dominant.
        - The 'New Id' assignments are completely independent of the original 'Unique Id' values.
            The original IDs are not used or preserved; the new IDs are based only on this function's logic.
        - If you want to keep the original IDs, use them directly without running this function.

        Example Usage
        -------------
        >>> df = ...  # Your YOLO tracking output as a DataFrame
        >>> result = reassign_ids_directional_cross_fix(df, distance_threshold=0.04, yolo_ids=[0])
        >>> print(result[['Frame Count', 'YOLO_id', 'X-center', 'New Id']])
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
