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

        # Arrays to keep track state: track_id, x position, direction, last frame
        track_ids = np.array([], dtype=int)      # Unique ID for each track
        track_xs = np.array([], dtype=float)     # Last known X position
        track_dirs = np.array([], dtype=int)     # Direction: 0=unknown, 1=right, -1=left
        track_last_frames = np.array([], dtype=int)  # Last frame seen

        # Group detections by frame for sequential processing
        frame_groups = df.groupby('Frame Count')

        # Process each frame in order
        for frame, frame_detections in frame_groups:
            # Indices (row numbers) of detections in current frame
            det_indices = frame_detections.index.values

            # X positions of detections in current frame
            det_xs = frame_detections['X-center'].values
            n_tracks = len(track_ids)  # How many active tracks exist now?
            n_dets = len(det_xs)       # How many detections in this frame?

            assigned_tracks = set()  # Which tracks have been matched this frame
            assigned_dets = set()    # Which detections have already been matched

            # If there are any tracks and detections, try to match
            if n_tracks > 0 and n_dets > 0:
                # Calculate (tracks x detections) difference matrix
                dx = det_xs[None, :] - track_xs[:, None]   # Shape: (n_tracks, n_dets)
                dists = np.abs(dx)                         # Distance between each track and detection

                # Compute movement direction for each possible match
                # np.sign(dx): -1 for left, 1 for right, 0 if same position
                new_dirs = np.sign(dx).astype(int)

                # Check if matching is allowed:
                # Allowed if track direction is unknown (0), or matches new direction
                dir_mask = (track_dirs[:, None] == 0) | (track_dirs[:, None] == new_dirs)

                # Only match if within distance threshold and direction is consistent
                can_match = (dists < distance_threshold) & dir_mask

                # Try to assign detections to the nearest allowed track (greedy)
                for det_idx in range(n_dets):
                    # Which tracks could match this detection?
                    candidates = np.where(can_match[:, det_idx])[0]
                    if candidates.size > 0:
                        # Pick the closest track among candidates
                        best_track_idx = candidates[np.argmin(dists[candidates, det_idx])]
                        if best_track_idx not in assigned_tracks:

                            # Assign detection to this track
                            df.at[det_indices[det_idx], 'New Id'] = track_ids[best_track_idx]

                            # Update this track's info: new position, direction, frame
                            # If new direction is 0 (no movement), keep old direction
                            new_dir = new_dirs[best_track_idx, det_idx]
                            track_dirs[best_track_idx] = new_dir if new_dir != 0 else track_dirs[best_track_idx]
                            track_xs[best_track_idx] = det_xs[det_idx]
                            track_last_frames[best_track_idx] = frame
                            assigned_tracks.add(best_track_idx)
                            assigned_dets.add(det_idx)

            # For any detection not assigned, create a new track/ID
            for det_idx in range(n_dets):
                if det_idx not in assigned_dets:
                    df.at[det_indices[det_idx], 'New Id'] = next_id
                    # Add new track info to arrays
                    track_ids = np.append(track_ids, next_id)
                    track_xs = np.append(track_xs, det_xs[det_idx])
                    track_dirs = np.append(track_dirs, 0)  # Start with unknown direction
                    track_last_frames = np.append(track_last_frames, frame)
                    next_id += 1  # Increment for the next new track

        # All frames processed, DataFrame now has 'New Id' for each detection
        return df
