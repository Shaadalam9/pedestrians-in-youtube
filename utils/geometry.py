# by Shadab Alam <md_shadab_alam@outlook.com>
import numpy as np
import pandas as pd
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

    def reassign_ids_directional_cross_fix(self, df, distance_threshold=0.05, yolo_ids=None, history_frames=3):
        """
        Reassigns object tracking IDs to correct for identity switches (ID swaps)
        when objects cross paths (even in same or opposite directions, e.g., overtaking),
        using consistent X-axis movement, short-term movement history, and optional class filtering.

        This function post-processes YOLO (or similar) tracking outputs in video data,
        where objects may switch tracking IDs after crossing or overtaking each other.
        The algorithm matches detections across frames based on spatial proximity along
        the X-axis, recent movement trajectory, and maintains consistent movement patterns
        to reduce erroneous ID swaps. Each detection is assigned a new 'Unique Id'
        that reflects the real-world object more consistently over time.

        Main Features
        -------------
        - Processes all detections, but applies the fix only to specified YOLO classes (`yolo_ids`).
        - Tracks are matched by X-center position, movement direction, and short-term X history.
        - Overtaking (objects moving in the same direction but swapping positions/IDs) is handled
            using multi-frame X trajectory matching.
        - All other (non-target) classes are preserved unchanged, and their IDs are never altered.

        Parameters
        ----------
        df : pandas.DataFrame
            Must include columns:
                - 'YOLO_id'    : YOLO class ID for each detection.
                - 'X-center'   : Normalized X center of bounding box (0 to 1).
                - 'Unique Id'  : YOLO's original object/tracking ID for each detection.
                - 'Frame Count': Frame number (sortable in time order).
            All other columns are preserved.

        distance_threshold : float, optional
            Maximum allowed normalized X distance (0 to 1) to associate a detection with an existing track.
            (Default: 0.05, i.e., detections within 5% of frame width can be matched).

        yolo_ids : list of int or None, optional
            YOLO class IDs to apply the fix to. Others remain unchanged. If None, all detected classes are fixed.

        history_frames : int, optional
            Number of previous frames to use for X-trajectory matching during assignment.
            Higher values help with longer, smoother trajectories but may fail if objects change direction often.

        Returns
        -------
        pandas.DataFrame
            Copy of the input DataFrame with:
                - 'Unique Id'      : The **corrected** identity for each detection (stable over time).
                - 'old_unique_id'  : The original YOLO 'Unique Id' for reference.
            All other columns are preserved.

        Limitations & Notes
        -------------------
        - This method assumes objects mostly move along the X-axis and their movement is smooth.
        - For non-target YOLO classes, no corrections are applied; their 'Unique Id' remains unchanged.
        - The corrected 'Unique Id' for each object remains constant across frames, even if YOLO swaps IDs
            during crossing or overtaking.

        Example Usage
        -------------
        >>> df = ...  # Your YOLO tracking output as a DataFrame
        >>> result = reassign_ids_directional_cross_fix(df, distance_threshold=0.04, yolo_ids=[0])
        >>> print(result[['Frame Count', 'YOLO_id', 'X-center', 'Unique Id', 'old_unique_id']])
        """

        # --- STEP 1: INITIALISE OUTPUT AND PROCESSING MASK ---

        # Work on a copy of the DataFrame to avoid mutating user data
        df = df.copy()

        # Initialise new column 'New Id' as the original YOLO Unique Id; will be replaced for fixed classes
        df['New Id'] = df['Unique Id']

        # Build a boolean mask: True for rows to fix (in yolo_ids), False for rows to leave untouched
        if yolo_ids is not None:
            mask = df['YOLO_id'].isin(yolo_ids)
        else:
            mask = pd.Series(True, index=df.index)  # If None, fix all classes

        # Select and sort only the relevant subset (target classes), ordered by frame
        df_fix = df[mask].sort_values(by='Frame Count').copy()

        # Set placeholder value for the new IDs in these rows
        df_fix['New Id'] = -1

        # --- STEP 2: PRECOMPUTE DETECTION X-HISTORY DICTIONARY ---

        # This dictionary allows instant O(1) lookup of a detection's X-center by (Unique Id, Frame Count)
        # Example: det_hist_dict[(42, 100)] returns the X-center for Unique Id 42 in frame 100
        det_hist_dict = {
            (row['Unique Id'], row['Frame Count']): row['X-center']
            for idx, row in df_fix.iterrows()
        }

        # --- STEP 3: TRACK MANAGEMENT SETUP ---

        # List of all active tracks. Each track is a dict:
        #  - 'unique_id':  the original Unique Id (which is stable and used as the corrected ID)
        #  - 'xs':         list of recent X-center positions for trajectory comparison
        #  - 'frames':     list of corresponding frame numbers for those Xs
        tracks = []

        # Optional: mapping from unique_id to track index (could help with custom matching logic)
        track_id_map = {}

        # --- STEP 4: MAIN FRAMEWISE ASSOCIATION LOOP ---

        # Go through frames in chronological order for correct temporal association
        frame_numbers = sorted(df_fix['Frame Count'].unique())

        for frame in frame_numbers:
            # --- 4A: GET DETECTIONS IN THIS FRAME ---
            detections = df_fix[df_fix['Frame Count'] == frame]
            det_indices = detections.index.values        # Row indices in df_fix for current frame
            det_xs = detections['X-center'].values       # X-center for each detection in this frame
            det_unique_ids = detections['Unique Id'].values  # Original YOLO Unique Id for each detection
            n_dets = len(det_xs)

            # Sets to keep track of which tracks/detections have already been matched in this frame
            assigned_tracks = set()
            assigned_dets = set()

            # --- 4B: MATCH EXISTING TRACKS TO DETECTIONS (TRAJECTORY ASSOCIATION) ---
            if tracks:  # Only run if we have existing tracks to match to

                # Array of each track's last X-center (to compute proximity quickly)
                last_xs = np.array([track['xs'][-1] for track in tracks])

                # Compute absolute X difference for all track/detection pairs
                # dists[i, j] = |last_xs[i] - det_xs[j]|
                dists = np.abs(last_xs[:, None] - det_xs[None, :])

                # Loop over each detection to find the best track
                for det_idx in range(n_dets):
                    best_track = None       # Index of best-matching track
                    best_score = float('inf')  # Lower = better match
                    this_uid = det_unique_ids[det_idx]

                    # --- 4B1: BUILD SHORT-TERM X-TRAJECTORY FOR THIS DETECTION ---
                    det_hist = []

                    # Look back up to history_frames in time (including this frame)
                    for back in range(history_frames-1, -1, -1):
                        key = (this_uid, frame - back)
                        if key in det_hist_dict:
                            # Append that frame's X-center if present
                            det_hist.append(det_hist_dict[key])
                    # det_hist: [x(t-n), ..., x(t-1), x(t)]

                    # --- 4B2: COMPARE TO EACH TRACK ---
                    for t_idx, track in enumerate(tracks):
                        if t_idx in assigned_tracks:
                            continue  # Skip if this track has already been matched this frame

                        # Only consider tracks close in X-center for this frame
                        if dists[t_idx, det_idx] >= distance_threshold:
                            continue

                        # Extract the last len(det_hist) Xs from this track for fair comparison
                        track_hist = track['xs'][-len(det_hist):] if len(det_hist) > 0 else []

                        # Compute trajectory score (mean absolute diff), if both histories are long enough
                        if len(det_hist) > 1 and len(track_hist) == len(det_hist):
                            traj_score = np.mean(np.abs(np.array(det_hist) - np.array(track_hist)))
                        elif len(track_hist) > 0:
                            traj_score = abs(track_hist[-1] - det_xs[det_idx])
                        else:
                            traj_score = dists[t_idx, det_idx]

                        # Greedy: assign to the best match (lowest score)
                        if traj_score < best_score:
                            best_score = traj_score
                            best_track = t_idx

                    # --- 4B3: ASSIGN DETECTION TO BEST TRACK IF FOUND ---
                    if best_track is not None and best_track not in assigned_tracks:
                        # Update 'New Id' for this detection to the track's unique_id (which is the original identity)
                        df_fix.at[det_indices[det_idx], 'New Id'] = tracks[best_track]['unique_id']

                        # Update the matched track's history with this detection's data
                        tracks[best_track]['xs'].append(det_xs[det_idx])
                        tracks[best_track]['frames'].append(frame)

                        # Only keep the last 'history_frames' points to keep memory bounded
                        if len(tracks[best_track]['xs']) > history_frames:
                            tracks[best_track]['xs'] = tracks[best_track]['xs'][-history_frames:]
                            tracks[best_track]['frames'] = tracks[best_track]['frames'][-history_frames:]

                        # Mark this track/detection as matched for this frame
                        assigned_tracks.add(best_track)
                        assigned_dets.add(det_idx)

            # --- 4C: CREATE NEW TRACKS FOR UNMATCHED DETECTIONS ---
            for det_idx in range(n_dets):
                if det_idx not in assigned_dets:
                    # This detection did not match any existing track, so create a new track for it
                    this_uid = det_unique_ids[det_idx]
                    df_fix.at[det_indices[det_idx], 'New Id'] = this_uid  # Use the detection's original Unique Id
                    track = {
                        'unique_id': this_uid,
                        'xs': [det_xs[det_idx]],  # Start new X history
                        'frames': [frame]         # Start new frame history
                    }
                    tracks.append(track)
                    track_id_map[this_uid] = len(tracks) - 1  # Optional, in case you want O(1) lookup later

        # --- STEP 5: COPY CORRECTED IDS INTO THE ORIGINAL DATAFRAME ---

        # Only update the processed rows (the target YOLO classes) with their new IDs
        df.loc[df_fix.index, 'New Id'] = df_fix['New Id']

        # --- STEP 6: FINAL COLUMN RENAMING FOR OUTPUT ---

        # 1. Safety: Any unmatched detection gets its original 'Unique Id'
        missing_newid_mask = df_fix['New Id'].isnull() | (df_fix['New Id'] == -1)
        df_fix.loc[missing_newid_mask, 'New Id'] = df_fix.loc[missing_newid_mask, 'Unique Id']

        # 2. Copy corrected IDs back to main DataFrame for processed rows
        df.loc[df_fix.index, 'New Id'] = df_fix['New Id']

        # Rename the original YOLO 'Unique Id' to 'old_unique_id' (for reference),
        # and the fixed, stable IDs to 'Unique Id'
        df = df.rename(columns={'Unique Id': 'old_unique_id', 'New Id': 'Unique Id'})

        return df
