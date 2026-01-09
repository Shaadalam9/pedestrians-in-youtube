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
        avg_height,                      # ALWAYS in centimeters (e.g., 170, 171)
        min_shared_frames: int = 4,      # keep 4 to match "working all good" behavior

        # Pixel/cm-mode thresholds (used when we have pixels, i.e., pixel CSV or img_w/img_h provided)
        dist_thresh: float = 80,         # centimeters
        similarity_thresh: float = 0.8,
        overlap_ratio: float = 0.7,

        # Normalized-mode tuned thresholds (used when normalized and no img_w/img_h)
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

        # OPTIONAL: if coords are normalized and we pass frame size, we can convert to pixels and use cm logic
        img_w: int | None = None,
        img_h: int | None = None,

        eps: float = 1e-9,
    ) -> bool:
        """Determines whether a tracked person is a rider (bicycle or motorcycle).

        This function classifies a YOLO-tracked `unique-id` (person) as a *rider* if it
        co-occurs with a bicycle or motorcycle track and the two tracks exhibit evidence
        of moving together.

        The implementation supports two coordinate representations commonly used in YOLO
        exports:

        1) Pixel coordinates:
           - `x-center`, `y-center`, `width`, `height` are in pixels.
           - In this mode, the function can use `avg_height` (centimeters) to approximate
             a per-frame pixel-to-centimeter scale and compare distances against
             `dist_thresh` (centimeters).

        2) Normalized coordinates:
           - `x-center`, `y-center`, `width`, `height` are in [0, 1] fractions of frame size.
           - In this mode, pixel-to-centimeter scaling is not possible unless the actual
             frame dimensions are provided (`img_w`, `img_h`).
           - If `img_w` and `img_h` are not provided, the function uses a scale-free
             normalized method based on distances relative to person height and a spatial
             configuration model that is robust to zoom and resolution changes.

        Mode selection:
          A) Pixel CSV (not normalized):
             Uses centimeter distance logic (original behavior) based on avg_height_cm.

          B) Normalized CSV + (img_w, img_h) provided:
             Converts normalized coordinates to pixels, then uses centimeter distance logic.

          C) Normalized CSV + no frame size:
             Uses tuned normalized logic (distance relative to person height + spatial/motion
             gating + short-overlap guard).

        The function returns True as soon as it finds *any* vehicle track that sufficiently
        matches the person track under the chosen mode.

        Args:
          df: Polars DataFrame containing YOLO detections with (at minimum) the columns:
            - yolo-id: YOLO class id (0=person, 1=bicycle, 3=motorcycle).
            - unique-id: tracker id for a detection.
            - frame-count: integer frame index within the CSV segment.
            - x-center, y-center, width, height: box geometry (pixel or normalized).
            - confidence: detection confidence (optional, used for de-duplication).
          id: Unique tracker id of the person to evaluate (a `unique-id` where yolo-id == 0).
          avg_height: Average real-world person height in centimeters (e.g., 170, 171). This is
            always treated as centimeters even in normalized mode (it is simply not used for
            scaling unless pixels are available).
          min_shared_frames: Minimum number of shared frames required between person and a vehicle
            track before attempting to classify. This is a recall/precision control; lower values
            increase recall and can increase false positives for brief co-location.
          dist_thresh: (Pixel/cm modes only) Maximum allowed person↔vehicle center distance in
            centimeters for a frame to count as “close enough.”
          similarity_thresh: (Pixel/cm modes only) Minimum cosine similarity between per-frame motion
            vectors for a step to count as “moving together.”
          overlap_ratio: (Pixel/cm modes only) Fraction of frames/steps that must satisfy proximity
            and motion similarity to accept the pair.
          dist_rel_thresh: (Normalized mode only) Maximum allowed center distance divided by person
            height (dimensionless). Example: 0.8 means “within 0.8 person-heights.”
          prox_req: (Normalized mode only) Minimum fraction of frames satisfying dist_rel_thresh.
          alpha_x: (Normalized mode only) Lateral gating: requires |dx| < alpha_x * person_width.
          beta_y: (Normalized mode only) Vertical gating (lower bound): requires dy > beta_y * person_height
            so the vehicle is below the person center.
          gamma_y: (Normalized mode only) Vertical gating (upper bound): requires dy < gamma_y * person_height
            so the vehicle is not unreasonably far below the person.
          coloc_req: (Normalized mode only) Minimum fraction of frames satisfying both proximity and the
            spatial configuration constraints (co-location ratio).
          sim_thresh: (Normalized mode only) Motion similarity threshold used when computing `sim_ratio`.
          sim_req: (Normalized mode only) Motion fallback accept threshold on `sim_ratio`.
          min_motion_steps: (Normalized mode only) Minimum number of proximate, non-zero-motion steps
            needed before trusting the motion similarity ratio.
          motion_coloc_min: (Normalized mode only) When accepting via the motion fallback, require at
            least this co-location ratio to avoid “walking alongside a bike” false positives.
          short_shared_frames: (Normalized mode only) Treat pairs with fewer shared frames than this
            as “short overlaps” that need extra evidence.
          short_sim_req: (Normalized mode only) For short overlaps, require very high motion consistency
            unless the person displacement condition is met.
          short_disp_req: (Normalized mode only) For short overlaps, allow acceptance if the person’s
            displacement (relative to their height) is large enough, even if motion similarity is not.
          img_w: Optional frame width in pixels (required to convert normalized x/width to pixels).
          img_h: Optional frame height in pixels (required to convert normalized y/height to pixels).
          eps: Small constant to avoid divide-by-zero and numerical instability.

        Returns:
          True if the person track is classified as a bicycle/motorcycle rider; otherwise False.

        Notes:
          - This function assumes that `unique-id` values are unique *within a class* but may collide
            across classes. To prevent a collision from corrupting results, we explicitly constrain the
            person track to yolo-id == 0.
          - Duplicates for the same (yolo-id, unique-id, frame-count) are handled by de-duplicating to
            the highest-confidence row before analysis.
          - The normalized-mode logic is generally more robust across videos because it does not depend
            on camera calibration or absolute pixel scale.
        """

        # -------------------------------------------------------------------------
        # Defensive input validation:
        # - avg_height is always interpreted as centimeters.
        # - We must guard against None, strings, and invalid values early to avoid
        #   cryptic runtime warnings from numpy divisions.
        # -------------------------------------------------------------------------
        try:
            if avg_height is None or float(avg_height) <= 0.0:
                return False
            avg_height_cm = float(avg_height)
        except Exception:
            return False

        # -------------------------------------------------------------------------
        # De-duplicate detections:
        # - Many detectors/exporters can emit multiple detections for the same
        #   (class, track-id, frame) due to post-processing quirks.
        # - Keeping the highest-confidence row stabilizes the join alignment and
        #   reduces jitter in position and size.
        # -------------------------------------------------------------------------
        df = Detection._dedup_per_frame(df)

        # -------------------------------------------------------------------------
        # Determine coordinate system:
        # - Normalized YOLO: values typically in [0, 1].
        # - Pixel YOLO: values in tens/hundreds of pixels.
        #
        # We use a simple heuristic: if max(x,y,w,h) <= ~1.5 on a sample, treat as normalized.
        # If anything goes wrong, default to normalized (safer for preventing cm misuse).
        # -------------------------------------------------------------------------
        try:
            sample = df.select(
                [
                    pl.col("x-center").cast(pl.Float64),
                    pl.col("y-center").cast(pl.Float64),
                    pl.col("width").cast(pl.Float64),
                    pl.col("height").cast(pl.Float64),
                ]
            ).head(5000)
            normalized = sample.height == 0 or float(np.nanmax(sample.to_numpy())) <= 1.5
        except Exception:
            normalized = True

        # -------------------------------------------------------------------------
        # Extract the person track:
        # - We explicitly filter yolo-id == 0 to ensure we only evaluate a person.
        # - If the person is absent (no rows), the id cannot be a rider.
        # -------------------------------------------------------------------------
        person_track = (
            df.filter((pl.col("yolo-id") == 0) & (pl.col("unique-id") == id))
            .sort("frame-count")
        )
        if person_track.height == 0:
            return False

        # -------------------------------------------------------------------------
        # The person must exist for enough frames to make a meaningful decision.
        # This prevents extremely short tracks from producing unstable classifications.
        # -------------------------------------------------------------------------
        frames = person_track.get_column("frame-count").to_numpy()
        if frames.size < min_shared_frames:
            return False

        # -------------------------------------------------------------------------
        # Restrict the candidate vehicle search window:
        # - Only consider bicycle/motorcycle detections during the timeframe where the
        #   person is visible. This reduces the candidate set and avoids pointless joins.
        # -------------------------------------------------------------------------
        first_frame = int(frames.min())
        last_frame = int(frames.max())

        vehicles_in_frames = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in([1, 3]))
        )
        if vehicles_in_frames.height == 0:
            return False

        # -------------------------------------------------------------------------
        # Collect vehicle track ids seen in the person’s time window.
        # We will try matching the person against each vehicle and accept on the first
        # sufficiently good match.
        # -------------------------------------------------------------------------
        vehicle_ids = vehicles_in_frames.select("unique-id").unique().to_series().to_list()

        # -------------------------------------------------------------------------
        # To align person and vehicle trajectories, we join on `frame-count`.
        # This is more reliable than truncating two sorted arrays because it keeps
        # exact temporal correspondence.
        # -------------------------------------------------------------------------
        p1 = person_track.unique(subset=["frame-count"], keep="first")

        # -------------------------------------------------------------------------
        # If the data is normalized but we also know the frame size, we can recover
        # pixel units:
        # - Convert centers to pixels using (img_w, img_h).
        # - Convert normalized person height to pixel height using img_h.
        # This enables centimeter scaling using avg_height_cm.
        # -------------------------------------------------------------------------
        can_use_cm_with_norm = (
            normalized
            and (img_w is not None)
            and (img_h is not None)
            and (img_w > 0)
            and (img_h > 0)
        )

        # =========================================================================
        # MODE A/B: CENTIMETER-BASED SCALING (pixel coords OR normalized+frame size)
        # =========================================================================
        if (not normalized) or can_use_cm_with_norm:
            # ---------------------------------------------------------------------
            # Helper: convert a (n,2) array of normalized xy coordinates to pixels.
            # We keep this as a nested function because it is only relevant here.
            # ---------------------------------------------------------------------
            def to_px_xy(xy_norm: np.ndarray) -> np.ndarray:
                out = xy_norm.copy()
                out[:, 0] *= float(img_w)  # type: ignore[arg-type]  # x scales with width
                out[:, 1] *= float(img_h)  # type: ignore[arg-type]  # y scales with height
                return out

            # ---------------------------------------------------------------------
            # For each candidate vehicle track:
            # - Join with the person track on frame-count.
            # - Compute a per-frame person↔vehicle center distance (in cm).
            # - Compute per-step motion similarity (cosine similarity).
            # - Require a sufficient fraction of overlap frames/steps to pass gates.
            # ---------------------------------------------------------------------
            for vehicle_id in vehicle_ids:
                v_track = vehicles_in_frames.filter(pl.col("unique-id") == vehicle_id).sort("frame-count")
                if v_track.height == 0:
                    continue

                v1 = v_track.unique(subset=["frame-count"], keep="first")
                j = p1.join(v1, on="frame-count", how="inner", suffix="_v")
                shared = j.height
                if shared < min_shared_frames:
                    continue

                # Extract aligned trajectories (center positions).
                p_xy = j.select(["x-center", "y-center"]).to_numpy()
                v_xy = j.select(["x-center_v", "y-center_v"]).to_numpy()

                # Person height drives pixel→cm scaling.
                # In pixel mode: `height` is already pixels.
                # In normalized+frame size mode: multiply by img_h to obtain pixels.
                p_h = j.get_column("height").to_numpy()
                if can_use_cm_with_norm:
                    p_xy = to_px_xy(p_xy)
                    v_xy = to_px_xy(v_xy)
                    p_h = p_h * float(img_h)  # type: ignore[arg-type]

                # Convert pixels to centimeters using an approximate scale:
                # px_per_cm = (person_height_px) / (avg_height_cm).
                # This treats the person box height as a proxy for real-world height.
                px_per_cm = p_h / avg_height_cm
                if np.any(px_per_cm <= 0):
                    continue

                # Per-frame distance between centers (cm).
                pixel_dists = np.linalg.norm(p_xy - v_xy, axis=1)
                dist_cm = pixel_dists / px_per_cm

                # Gate 1: proximity on a per-frame basis.
                proximity = dist_cm < dist_thresh
                if proximity.mean() < overlap_ratio:
                    continue

                # Per-step motion vectors (frame i -> i+1).
                p_mov = np.diff(p_xy, axis=0)
                v_mov = np.diff(v_xy, axis=0)
                if p_mov.shape[0] == 0:
                    continue

                # Cosine similarity of motion direction.
                na = np.linalg.norm(p_mov, axis=1)
                nb = np.linalg.norm(v_mov, axis=1)
                sim = np.zeros_like(na, dtype=float)
                mask = (na > eps) & (nb > eps)
                sim[mask] = (p_mov[mask] * v_mov[mask]).sum(axis=1) / (na[mask] * nb[mask])

                # Step gating: a step is considered “good” only if the step’s endpoint
                # frame is proximate (prox_steps) and the motion similarity is high.
                prox_steps = proximity[1:]
                m = min(len(prox_steps), len(sim))
                prox_steps = prox_steps[:m]
                sim = sim[:m]

                good_steps = (sim > similarity_thresh) & prox_steps

                # Gate 2: sufficient fraction of “good” steps.
                if good_steps.sum() / max(1, len(sim)) >= overlap_ratio:
                    return True

            # If no vehicle matched strongly enough, treat as non-rider.
            return False

        # =========================================================================
        # MODE C: NORMALIZED TUNED LOGIC (scale-free; no pixel/cm conversion)
        # =========================================================================
        #
        # This is the “working all good” rider logic for normalized YOLO coordinates:
        # - Proximity uses dist_rel = distance / person_height (dimensionless).
        # - Co-location includes a spatial model: vehicle should be below the person and
        #   laterally close.
        # - Motion similarity is an optional corroboration mechanism.
        # - Short-overlap guard prevents brief, accidental co-location false positives.
        # =========================================================================
        for vehicle_id in vehicle_ids:
            v_track = vehicles_in_frames.filter(pl.col("unique-id") == vehicle_id).sort("frame-count")
            if v_track.height == 0:
                continue

            v1 = v_track.unique(subset=["frame-count"], keep="first")
            j = p1.join(v1, on="frame-count", how="inner", suffix="_v")
            shared = j.height
            if shared < min_shared_frames:
                continue

            # Aligned centers (normalized units).
            p_xy = j.select(["x-center", "y-center"]).to_numpy()
            v_xy = j.select(["x-center_v", "y-center_v"]).to_numpy()

            # Person size (normalized); used to normalize distance and apply spatial gating.
            p_w = j.get_column("width").to_numpy()
            p_h = j.get_column("height").to_numpy()

            # Relative position of vehicle with respect to person (normalized units).
            relx = v_xy[:, 0] - p_xy[:, 0]
            rely = v_xy[:, 1] - p_xy[:, 1]

            # Distance normalized by person height (dimensionless).
            dist = np.linalg.norm(p_xy - v_xy, axis=1)
            dist_rel = dist / np.maximum(p_h, eps)

            # Gate 1: proximity ratio must be high enough.
            prox = dist_rel < dist_rel_thresh
            if prox.mean() < prox_req:
                continue

            # Gate 2: spatial configuration gating.
            # The bike/motorcycle box center is expected to be below the person’s center
            # and not too far sideways.
            spatial = (
                (np.abs(relx) < alpha_x * p_w)
                & (rely > beta_y * p_h)
                & (rely < gamma_y * p_h)
            )
            coloc = prox & spatial
            coloc_ratio = float(coloc.mean())

            # Optional motion similarity:
            # - Only computed on frames where both tracks move (non-zero motion).
            # - Only computed over proximate steps to reduce spurious correlations.
            p_mov = np.diff(p_xy, axis=0)
            v_mov = np.diff(v_xy, axis=0)
            sim_ratio = 0.0

            if p_mov.shape[0] > 0:
                na = np.linalg.norm(p_mov, axis=1)
                nb = np.linalg.norm(v_mov, axis=1)
                move_mask = (na > eps) & (nb > eps)

                cos = np.zeros_like(na, dtype=float)
                cos[move_mask] = (p_mov[move_mask] * v_mov[move_mask]).sum(axis=1) / (na[move_mask] * nb[move_mask])

                prox_steps = prox[1:]
                m = min(len(prox_steps), len(cos))
                prox_steps = prox_steps[:m]
                cos = cos[:m]
                move_mask = move_mask[:m]

                denom_mask = prox_steps & move_mask
                denom = int(denom_mask.sum())

                # Do not trust motion similarity if we have too few valid steps.
                if denom >= min_motion_steps:
                    sim_ratio = float(((cos > sim_thresh) & denom_mask).sum() / denom)

            # Short-overlap guard:
            # - If the overlap is short, a person can briefly pass near a bicycle and
            #   appear “close” for a handful of frames.
            # - We require either strong motion similarity or a meaningful displacement.
            if shared < short_shared_frames:
                if shared > 1:
                    p_disp = float(np.linalg.norm(p_xy[-1] - p_xy[0]))
                    p_disp_rel = p_disp / float(np.maximum(np.mean(p_h), eps))
                else:
                    p_disp_rel = 0.0

                if not (sim_ratio >= short_sim_req or p_disp_rel >= short_disp_req):
                    continue

            # Final accept logic:
            # - Strong co-location is sufficient by itself.
            # - Alternatively, strong motion similarity can accept, but only if co-location
            #   is not too low (avoids “walking alongside a bike” false positives).
            ok = (coloc_ratio >= coloc_req) or (sim_ratio >= sim_req and coloc_ratio >= motion_coloc_min)
            if ok:
                return True

        # No vehicle track matched the person strongly enough.
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
