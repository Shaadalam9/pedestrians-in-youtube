import numpy as np
import polars as pl
from utils.core.metadata import MetaData
from helper_script import Youtube_Helper

metadata = MetaData()
helper = Youtube_Helper()


class Detection:

    def __init__(self) -> None:
        pass

    def pedestrian_crossing(self, dataframe, video_id, df_mapping, min_x, max_x, person_id):
        """Counts the number of person with a specific ID crosses the road within specified boundaries.

        Args:
            dataframe (DataFrame): DataFrame containing data from the video.
            video_id (string): "video_start_index_fps
            min_x (float): Min/Max x-coordinate boundary for the road crossing.
            max_x (float): Max/Min x-coordinate boundary for the road crossing.
            person_id (int): Unique ID assigned by the YOLO tracker to identify the person.

        Returns:
            Tuple[int, list]: A tuple containing the number of person crossed the road within
            the boundaries and a list of unique IDs of the person.
        """

        # Filter dataframe to include only entries for the specified person
        crossed_df = dataframe.filter(pl.col("yolo-id") == person_id)

        if crossed_df.height == 0:
            return [], []

        # Group by unique-id and keep only those that cross both boundaries:
        # (x-center <= min_x).any() and (x-center >= max_x).any()
        # Equivalent to: min(x-center) <= min_x AND max(x-center) >= max_x
        crossed_ids = (crossed_df.group_by("unique-id").agg([
                pl.col("x-center").min().alias("_x_min"),
                pl.col("x-center").max().alias("_x_max"),
            ])
            .filter((pl.col("_x_min") <= min_x) & (pl.col("_x_max") >= max_x))
            .select("unique-id").to_series().to_list()
        )

        # Lookup avg_height via existing MetaData helper (convert mapping once if needed)
        avg_height = None

        # df_mapping_for_meta = df_mapping.to_pandas() if isinstance(df_mapping, pl.DataFrame) else df_mapping
        result = metadata.find_values_with_video_id(df_mapping, video_id)
        if result is not None:
            avg_height = result[15]

        pedestrian_ids = []
        for uid in crossed_ids:
            if Detection.is_rider_id(dataframe, uid, avg_height):
                continue  # Filter out riders
            if not self.is_valid_crossing(dataframe, uid):
                continue  # Skip fake crossing
            pedestrian_ids.append(uid)
        return pedestrian_ids, crossed_ids

    @staticmethod
    def _dedup_per_frame(df: pl.DataFrame) -> pl.DataFrame:
        """Keep highest-confidence detection per (yolo-id, unique-id, frame-count)."""
        if "confidence" not in df.columns:
            return df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

        return (
            df.sort(
                ["yolo-id", "unique-id", "frame-count", "confidence"],
                descending=[False, False, False, True],
            )
            .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
        )

    @staticmethod
    def is_rider_id(
        df: pl.DataFrame,
        id,
        avg_height,
        min_shared_frames: int = 4,
        # Normalized-mode tuned thresholds (YOLO coords in 0..1)
        dist_rel_thresh: float = 0.8,
        prox_req: float = 0.7,
        alpha_x: float = 1.0,
        beta_y: float = 0.03,
        gamma_y: float = 1.4,
        coloc_req: float = 0.7,
        sim_thresh: float = 0.4,
        sim_req: float = 0.5,
        min_motion_steps: int = 3,
        motion_coloc_min: float = 0.5,
        # Short-overlap guard (prevents brief accidental co-location)
        short_shared_frames: int = 8,
        short_sim_req: float = 0.8,
        short_disp_req: float = 0.12,
        eps: float = 1e-9,
    ) -> bool:
        """
        Returns whether a tracked person is a rider (bicycle or motorcycle) using normalized YOLO boxes.

        This implementation assumes the YOLO CSV geometry columns are normalized to the frame:
        `x-center`, `y-center`, `width`, `height` ∈ [0, 1]. In this representation, values are
        *fractions of the frame size*, not pixels. Therefore, it is not valid to compute real-world
        distances in centimeters directly from the CSV unless frame pixel dimensions and camera
        geometry are available.

        Instead, this method uses a *scale-free* criterion to detect person–vehicle pairs that move
        together:

        - **Proximity (scale-free):** The person and vehicle centers must remain close relative to
          the person's height:
            `dist_rel = ||p_center - v_center|| / p_height`.
          This ratio is dimensionless and robust to zoom/resolution.

        - **Spatial configuration:** For a rider, the bicycle/motorcycle is typically below the
          person's center and not too far sideways. We enforce lateral and vertical bounds based on
          the person's width/height.

        - **Motion corroboration (optional):** When co-location is not strong enough on its own, we
          compute a motion-consistency ratio (cosine similarity of step vectors) on proximate steps.

        - **Short-overlap guard:** For brief overlaps, accidental co-location can look like a rider
          (e.g., a pedestrian walking next to a parked or passing bicycle). For short overlaps we
          require either very strong motion consistency or sufficient person displacement.

        Note on `avg_height`:
          You may pass average height in centimeters (e.g., 170/171) because your upstream API expects
          it. In this normalized-only version it is **validated** (must be positive) but **not used**
          for any scaling, because centimeter scaling is not well-defined in normalized coordinates.

        Args:
          df (pl.DataFrame): YOLO detections containing at least:
            - `yolo-id`: class id (0=person, 1=bicycle, 3=motorcycle)
            - `unique-id`: tracker id
            - `frame-count`: frame index in the segment
            - `x-center`, `y-center`, `width`, `height`: normalized box geometry (0..1)
            - `confidence`: optional; used by your project de-dup routine
          id (Any): Person `unique-id` to classify.
          avg_height (float): Average person height in centimeters. Must be > 0, but not otherwise used.
          min_shared_frames (int): Minimum number of frames in which both the person and a candidate
            vehicle track are present (after frame alignment) to attempt classification.

          dist_rel_thresh (float): Max allowed (center distance / person height) per frame.
          prox_req (float): Minimum fraction of frames that must satisfy `dist_rel_thresh`.

          alpha_x (float): Lateral bound coefficient: `|dx| < alpha_x * person_width`.
          beta_y (float): Vertical lower bound coefficient: `dy > beta_y * person_height` (vehicle below person).
          gamma_y (float): Vertical upper bound coefficient: `dy < gamma_y * person_height` (not too far below).
          coloc_req (float): Minimum fraction of frames satisfying proximity AND spatial constraints.

          sim_thresh (float): Cosine similarity threshold for a step to be counted as “moving together”.
          sim_req (float): Motion similarity ratio required for motion-based acceptance.
          min_motion_steps (int): Minimum number of proximate, non-zero-motion steps needed before using motion ratio.
          motion_coloc_min (float): When accepting via motion fallback, require at least this co-location ratio
            to avoid “walking alongside a bike” false positives.

          short_shared_frames (int): If shared frames < this value, treat as short overlap and apply stricter gating.
          short_sim_req (float): For short overlaps, require sim_ratio >= this OR sufficient displacement.
          short_disp_req (float): For short overlaps, allow acceptance if person displacement / person height >= this.

          eps (float): Numerical stabilizer for divides and norm checks.

        Returns:
          bool: True if the person is classified as a rider (bicycle or motorcycle), else False.
        """

        # ---------------------------------------------------------------------
        # Validate the provided real-world height (cm).
        # We keep this check because your calling code always supplies it and
        # it prevents accidental None/0/invalid values from silently passing.
        # ---------------------------------------------------------------------
        try:
            if avg_height is None or float(avg_height) <= 0.0:
                return False
        except Exception:
            return False

        # ---------------------------------------------------------------------
        # De-duplicate per (yolo-id, unique-id, frame-count).
        # Your project already has a robust implementation; we call it to:
        #   - keep highest-confidence detection per frame
        #   - stabilize joins and reduce jitter
        # ---------------------------------------------------------------------
        df = Detection._dedup_per_frame(df)

        # ---------------------------------------------------------------------
        # Extract only the person's track (yolo-id == 0).
        # If no rows exist, the id is not present as a person and cannot be a rider.
        # ---------------------------------------------------------------------
        person_track = (
            df.filter((pl.col("yolo-id") == 0) & (pl.col("unique-id") == id))
            .sort("frame-count")
        )
        if person_track.height == 0:
            return False

        # ---------------------------------------------------------------------
        # Require the person to exist for a minimum duration.
        # This avoids classifying extremely short/noisy tracks.
        # ---------------------------------------------------------------------
        frames = person_track.get_column("frame-count").to_numpy()
        if frames.size < min_shared_frames:
            return False

        first_frame = int(frames.min())
        last_frame = int(frames.max())

        # ---------------------------------------------------------------------
        # Gather bicycles/motorcycles in the time window where the person exists.
        # We only need vehicles that overlap temporally with the person.
        # ---------------------------------------------------------------------
        vehicles_in_frames = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in([1, 3]))
        )
        if vehicles_in_frames.height == 0:
            return False

        vehicle_ids = vehicles_in_frames.select("unique-id").unique().to_series().to_list()

        # ---------------------------------------------------------------------
        # Frame-count alignment:
        # We join person/vehicle rows on frame-count to ensure each point in the
        # trajectory corresponds to the same video frame.
        # ---------------------------------------------------------------------
        p1 = person_track.unique(subset=["frame-count"], keep="first")

        # ---------------------------------------------------------------------
        # Evaluate each candidate vehicle track; return True on the first match.
        # ---------------------------------------------------------------------
        for vehicle_id in vehicle_ids:
            v_track = vehicles_in_frames.filter(pl.col("unique-id") == vehicle_id).sort("frame-count")
            if v_track.height == 0:
                continue

            v1 = v_track.unique(subset=["frame-count"], keep="first")

            # ---------------------------------------------------------------
            # Inner join gives only frames where BOTH person and vehicle exist.
            # ---------------------------------------------------------------
            j = p1.join(v1, on="frame-count", how="inner", suffix="_v")
            shared = j.height
            if shared < min_shared_frames:
                continue

            # ---------------------------------------------------------------
            # Extract aligned geometry in normalized units.
            # ---------------------------------------------------------------
            p_xy = j.select(["x-center", "y-center"]).to_numpy()
            v_xy = j.select(["x-center_v", "y-center_v"]).to_numpy()
            p_w = j.get_column("width").to_numpy()
            p_h = j.get_column("height").to_numpy()

            # ---------------------------------------------------------------
            # Scale-free proximity: distance normalized by person height.
            # A rider should have the vehicle near the person relative to their size.
            # ---------------------------------------------------------------
            dist = np.linalg.norm(p_xy - v_xy, axis=1)
            dist_rel = dist / np.maximum(p_h, eps)
            prox = dist_rel < dist_rel_thresh

            # Require sufficient fraction of proximate frames.
            if float(prox.mean()) < prox_req:
                continue

            # ---------------------------------------------------------------
            # Spatial configuration gating:
            # - Vehicle is expected below the person center (positive dy).
            # - Vehicle should not be too far sideways.
            # ---------------------------------------------------------------
            relx = v_xy[:, 0] - p_xy[:, 0]
            rely = v_xy[:, 1] - p_xy[:, 1]
            spatial = (np.abs(relx) < alpha_x * p_w) & (rely > beta_y * p_h) & (rely < gamma_y * p_h)

            coloc = prox & spatial
            coloc_ratio = float(coloc.mean())

            # ---------------------------------------------------------------
            # Motion corroboration (optional):
            # We compute cosine similarity of step vectors for proximate steps
            # where both tracks actually move (non-zero motion).
            # ---------------------------------------------------------------
            p_mov = np.diff(p_xy, axis=0)
            v_mov = np.diff(v_xy, axis=0)

            sim_ratio = 0.0
            if p_mov.shape[0] > 0:
                na = np.linalg.norm(p_mov, axis=1)
                nb = np.linalg.norm(v_mov, axis=1)
                move_mask = (na > eps) & (nb > eps)

                cos = np.zeros_like(na, dtype=float)
                cos[move_mask] = (p_mov[move_mask] * v_mov[move_mask]).sum(axis=1) / (na[move_mask] * nb[move_mask])

                # Step i corresponds to frames i and i+1; use proximity at frame i+1.
                prox_steps = prox[1:]

                # Be defensive about length mismatches.
                m = min(len(prox_steps), len(cos), len(move_mask))
                prox_steps = prox_steps[:m]
                cos = cos[:m]
                move_mask = move_mask[:m]

                denom_mask = prox_steps & move_mask
                denom = int(denom_mask.sum())
                if denom >= min_motion_steps:
                    sim_ratio = float(((cos > sim_thresh) & denom_mask).sum() / denom)

            # ---------------------------------------------------------------
            # Short-overlap guard:
            # If the overlap is short, co-location can be incidental. Require either:
            # - very high motion consistency, or
            # - sufficient person displacement relative to height.
            # ---------------------------------------------------------------
            if shared < short_shared_frames:
                if shared > 1:
                    p_disp = float(np.linalg.norm(p_xy[-1] - p_xy[0]))
                    p_disp_rel = p_disp / float(np.maximum(np.mean(p_h), eps))
                else:
                    p_disp_rel = 0.0

                if not (sim_ratio >= short_sim_req or p_disp_rel >= short_disp_req):
                    continue

            # ---------------------------------------------------------------
            # Final decision:
            # - Accept if co-location is strong.
            # - Or accept via motion fallback only if co-location isn't too weak.
            #   (This is what removed false positives like “walking alongside a bike”.)
            # ---------------------------------------------------------------
            ok = (coloc_ratio >= coloc_req) or (sim_ratio >= sim_req and coloc_ratio >= motion_coloc_min)
            if ok:
                return True

        # No vehicle match met the criteria => not a rider.
        return False

    def is_valid_crossing(self, df, person_id, ratio_thresh=0.2):
        """
        Determines if a detected pedestrian crossing is valid based on the relative movement
        between the person and static objects (traffic light or stop sign) in the YOLO output.

        The function assumes the camera (e.g., dashcam) may be moving. To account for this,
        it compares the y-direction movement of the static object(s) and the person. If the
        static object's y-movement is a significant fraction of the person's y-movement, it
        is likely due to camera movement rather than actual pedestrian crossing, and the
        crossing is classified as invalid.

        Args:
            df (pd.DataFrame): DataFrame containing YOLO detections with columns:
                'yolo-id', 'x-center', 'y-center', 'width', 'height', 'unique-id', 'frame-count'.
                All coordinates are normalized between 0 and 1.
            person_id (int or str): Unique Id of the person (pedestrian) to be analyzed.
            key: Placeholder for additional parameters (not used in this function).
            avg_height: Placeholder for additional parameters (not used in this function).
            fps: Placeholder for additional parameters (not used in this function).
            ratio_thresh (float, optional): Threshold ratio of static object y-movement to
                person's y-movement. Default is 0.2.

        Returns:
            bool: True if crossing is valid (not due to camera movement), False otherwise.
        """
        person_track = df.filter(pl.col("unique-id") == person_id)
        if person_track.height == 0:
            return False

        frames = person_track.get_column("frame-count").to_numpy()
        first_frame = int(frames.min())
        last_frame = int(frames.max())

        static_objs = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in([9, 11]))
        )

        # No static object present; cannot verify, assume valid
        if static_objs.height == 0:
            return True

        # Person y-movement
        person_y_max = person_track.select(pl.col("y-center").max()).item()
        person_y_min = person_track.select(pl.col("y-center").min()).item()
        try:
            person_y_movement = float(person_y_max) - float(person_y_min)
        except Exception:
            return False

        if person_y_movement == 0:
            return False

        # Max static object y-movement across static object tracks
        static_movement_max = (
            static_objs
            .group_by("unique-id")
            .agg((pl.col("y-center").max() - pl.col("y-center").min()).alias("y_movement"))
            .select(pl.col("y_movement").max())
            .item()
        )

        try:
            movement_ratio = float(static_movement_max) / person_y_movement
        except Exception:
            return False

        return movement_ratio <= ratio_thresh
