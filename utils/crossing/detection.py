import numpy as np
import polars as pl
from utils.core.metadata import MetaData
from helper_script import Youtube_Helper
from typing import Tuple, List, Any

metadata = MetaData()
helper = Youtube_Helper()


class Detection:

    def __init__(self) -> None:
        pass

    def pedestrian_crossing(self, dataframe: pl.DataFrame, video_id: str, df_mapping, min_x: float, max_x: float,
                            person_id, tol: float = 0.00, min_track_frames: int = 10, min_road_frames: int = 3
                            ) -> Tuple[List[Any], List[Any]]:
        """
        Identifies pedestrian tracks that satisfy a road-crossing criterion and filters false positives.

        The function performs two stages:

        1) Crossing-candidate detection (state machine on x-center):
           - The image is partitioned into three zones based on x-center:
               LEFT  : x <= min_x - tol
               ROAD  : min_x + tol <= x <= max_x - tol
               RIGHT : x >= max_x + tol
             Values in the buffer region between these thresholds retain the previous state
             to reduce boundary flicker due to detector jitter.

           - A track is considered a crossing candidate if it:
               (a) spends at least `min_road_frames` frames in ROAD, and
               (b) has evidence of a side-to-side transition with an intermediate ROAD phase
                   (LEFT → ROAD → RIGHT or RIGHT → ROAD → LEFT). Tracks that start inside ROAD
                   are handled by checking for opposite-side presence before and after ROAD frames.

        2) Candidate filtering:
           - Vehicle-associated persons are removed via `Detection.is_rider_id(...)`.
           - Camera-motion artifacts (e.g., turns) are removed via `self.is_valid_crossing(...)`.

        Args:
            dataframe: Polars DataFrame containing YOLO detections with normalized geometry.
                Required columns:
                  - "yolo-id", "unique-id", "frame-count", "x-center"
                Additional columns used by downstream filters may include:
                  - "y-center", "width", "height", "confidence"
            video_id: Identifier used to retrieve metadata from `df_mapping`.
            df_mapping: Lookup structure consumed by `metadata.find_values_with_video_id`.
            min_x: Left boundary of the road band in normalized coordinates [0, 1].
            max_x: Right boundary of the road band in normalized coordinates [0, 1].
            person_id: Present for interface compatibility; not used for selection in this implementation.
            tol: Boundary hysteresis tolerance. Larger values reduce flicker but may reduce sensitivity.
            min_track_frames: Minimum number of frames required for a track to be considered.
            min_road_frames: Minimum number of frames the track must spend inside the ROAD zone.

        Returns:
            A tuple of two lists:
              - pedestrian_ids: unique-ids classified as pedestrian crossings after filtering.
              - crossed_ids: unique-ids that satisfy the crossing-candidate state-machine criterion.
        """
        # Restrict processing to the person class (COCO person == 0).
        crossed_df = dataframe.filter(pl.col("yolo-id") == 0)
        if crossed_df.height == 0:
            return [], []

        # De-duplicate per (yolo-id, unique-id, frame-count) to stabilize per-frame state.
        crossed_df = Detection._dedup_per_frame(crossed_df)

        # Prepare track data: one row per frame per id, sorted for sequential processing.
        tracks = (
            crossed_df.select(["unique-id", "frame-count", "x-center"])
            .sort(["unique-id", "frame-count"])
        )
        if tracks.height == 0:
            return [], []

        uids = tracks.select("unique-id").unique().to_series().to_list()

        # State thresholds with hysteresis.
        left_hard = float(min_x) - float(tol)
        left_soft = float(min_x) + float(tol)
        right_soft = float(max_x) - float(tol)
        right_hard = float(max_x) + float(tol)

        crossed_ids: List[Any] = []

        for uid in uids:
            tr = tracks.filter(pl.col("unique-id") == uid).sort("frame-count")
            n = tr.height
            if n < int(min_track_frames):
                continue

            x = tr.get_column("x-center").to_numpy()
            if x.size == 0:
                continue

            # State encoding: 0=LEFT, 1=ROAD, 2=RIGHT.
            states = np.empty(x.size, dtype=np.int8)

            # Initialize state from the first observation (no hysteresis).
            x0 = float(x[0])
            if x0 < float(min_x):
                s = 0
            elif x0 > float(max_x):
                s = 2
            else:
                s = 1
            states[0] = s

            # State update with hysteresis/buffer behavior.
            for i in range(1, x.size):
                xi = float(x[i])

                if xi <= left_hard:
                    s = 0
                elif xi >= right_hard:
                    s = 2
                elif left_soft <= xi <= right_soft:
                    s = 1
                else:
                    # Buffer region: keep previous state to suppress flicker near boundaries.
                    s = s

                states[i] = s

            is_left = states == 0
            is_road = states == 1
            is_right = states == 2

            # Minimum presence in the road band.
            if int(is_road.sum()) < int(min_road_frames):
                continue

            # Presence before/after each index for left/right zones.
            left_before = np.maximum.accumulate(is_left)
            right_before = np.maximum.accumulate(is_right)
            left_after = np.maximum.accumulate(is_left[::-1])[::-1]
            right_after = np.maximum.accumulate(is_right[::-1])[::-1]

            # A crossing is indicated by a ROAD index that separates the two sides.
            crossing_mask = is_road & ((left_before & right_after) | (right_before & left_after))
            if bool(crossing_mask.any()):
                crossed_ids.append(uid)

        # Metadata lookup for average height (used by the existing rider filter).
        avg_height = None
        result = metadata.find_values_with_video_id(df_mapping, video_id)
        if result is not None:
            avg_height = result[15]

        pedestrian_ids: List[Any] = []
        for uid in crossed_ids:
            if Detection.is_rider_id(dataframe, uid, avg_height):
                continue
            if not Detection.is_valid_crossing(dataframe, uid):
                continue
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
        # Normalised-mode tuned thresholds (YOLO coords in 0..1)
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
        # ---------------------------------------------------------------------
        # vehicle association beyond bicycle/motorcycle
        # ---------------------------------------------------------------------
        open_rider_class_ids: Tuple[int, ...] = (1, 3),          # bicycle, motorcycle
        enclosed_vehicle_class_ids: Tuple[int, ...] = (2, 5, 7, 6),  # car, bus, truck, train (COCO)
        enclosed_prox_req: float = 0.7,      # fraction of frames where person is close to vehicle center
        enclosed_dist_rel_thresh: float = 0.9,  # dist / vehicle_height (dimensionless)
        enclosed_inside_req: float = 0.4,    # fraction of frames where person center is inside vehicle bbox
        enclosed_rel_stab_max: float = 0.10,  # max robust range of relative position (normalised by vehicle size)
        enclosed_iou_req: float = 0.02,      # optional low IoU gate (kept permissive; many person-in-car boxes are small)  # noqa: E501
        q: float = 0.05,                     # quantile for robust range (5%..95%)
    ) -> bool:
        """
        Returns whether a tracked person is a "rider / vehicle-associated person" using normalised YOLO boxes.

        What this function treats as "rider" (so we can exclude from pedestrian crossing):
          1) Open riders: person riding *bicycle/motorcycle* (existing logic).
          2) Enclosed vehicle association: person that appears *inside/attached to* a car/bus/truck/train.
             This catches cases where the detector outputs a 'person' for a rider/passenger/driver that
             moves with the vehicle, and we do not want to count them as a pedestrian crossing.

        Assumptions:
          - YOLO geometry columns are normalised to the frame:
            x-center, y-center, width, height ∈ [0, 1].
          - "frame-count" aligns detections across objects.

        How enclosed-vehicle association works:
          - The person's center should be inside the vehicle bbox for a meaningful fraction of shared frames.
          - The person's position relative to the vehicle should be stable over time:
              relx_norm = (x_person - x_vehicle) / max(vehicle_width, eps)
              rely_norm = (y_person - y_vehicle) / max(vehicle_height, eps)
            A true passenger/driver tends to have low variation in these values.
          - Proximity is computed scale-free using vehicle height:
              dist_rel = ||p_center - v_center|| / max(v_height, eps)

        Args:
          df: Polars DataFrame with YOLO detections (normalised coords).
          id: Person unique-id to classify.
          avg_height: Kept for compatibility; validated but not used in normalised-only logic.
          ... (existing params preserved)
          open_rider_class_ids: Vehicle classes that should use open-rider geometry checks.
          enclosed_vehicle_class_ids: Vehicle classes treated as enclosed vehicles.
          enclosed_*: Tunables for enclosed vehicle association.
          q: Quantile for robust ranges.

        Returns:
          True if the person is classified as riding/vehicle-associated; else False.
        """

        # ---------------------------------------------------------------------
        # Validate avg_height for compatibility with our upstream API.
        # (Not used for scaling in normalised-only logic.)
        # ---------------------------------------------------------------------
        try:
            if avg_height is None or float(avg_height) <= 0.0:
                return False
        except Exception:
            return False

        # ---------------------------------------------------------------------
        # Deduplicate per (yolo-id, unique-id, frame-count) to stabilise joins.
        # ---------------------------------------------------------------------
        df = Detection._dedup_per_frame(df)

        # ---------------------------------------------------------------------
        # Extract person track (yolo-id == 0).
        # ---------------------------------------------------------------------
        person_track = (
            df.filter((pl.col("yolo-id") == 0) & (pl.col("unique-id") == id))
            .sort("frame-count")
        )
        if person_track.height == 0:
            return False

        frames = person_track.get_column("frame-count").to_numpy()
        if frames.size < min_shared_frames:
            return False

        first_frame = int(frames.min())
        last_frame = int(frames.max())

        # ---------------------------------------------------------------------
        # Gather all vehicle detections that overlap with the person in time.
        # We include:
        #   - open riders (bicycle, motorcycle)
        #   - enclosed vehicles (car, bus, truck, train by default)
        # ---------------------------------------------------------------------
        vehicle_classes = tuple(open_rider_class_ids) + tuple(enclosed_vehicle_class_ids)

        vehicles_in_frames = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in(list(vehicle_classes)))
        )
        if vehicles_in_frames.height == 0:
            return False

        vehicle_ids = vehicles_in_frames.select("unique-id").unique().to_series().to_list()

        # One row per frame for person track (clean alignment).
        p1 = person_track.unique(subset=["frame-count"], keep="first")

        # ---------------------------------------------------------------------
        # Small helper: robust range (q..1-q) to reduce jitter/outliers.
        # ---------------------------------------------------------------------
        def robust_range_np(x: np.ndarray) -> float:
            if x.size == 0:
                return 0.0
            lo = np.quantile(x, q)
            hi = np.quantile(x, 1.0 - q)
            return float(hi - lo)

        # ---------------------------------------------------------------------
        # Helper: IoU in normalised coordinates for two aligned arrays of boxes.
        # Boxes are (xc, yc, w, h) arrays, all normalised.
        # ---------------------------------------------------------------------
        def iou_xywh_norm(px, py, pw, ph, vx, vy, vw, vh) -> np.ndarray:
            px1 = px - pw / 2.0
            py1 = py - ph / 2.0
            px2 = px + pw / 2.0
            py2 = py + ph / 2.0

            vx1 = vx - vw / 2.0
            vy1 = vy - vh / 2.0
            vx2 = vx + vw / 2.0
            vy2 = vy + vh / 2.0

            ix1 = np.maximum(px1, vx1)
            iy1 = np.maximum(py1, vy1)
            ix2 = np.minimum(px2, vx2)
            iy2 = np.minimum(py2, vy2)

            iw = np.maximum(0.0, ix2 - ix1)
            ih = np.maximum(0.0, iy2 - iy1)
            inter = iw * ih

            p_area = np.maximum(0.0, px2 - px1) * np.maximum(0.0, py2 - py1)
            v_area = np.maximum(0.0, vx2 - vx1) * np.maximum(0.0, vy2 - vy1)
            union = np.maximum(p_area + v_area - inter, eps)

            return inter / union

        # ---------------------------------------------------------------------
        # Evaluate each candidate vehicle track; return True on the first match.
        # ---------------------------------------------------------------------
        for vehicle_id in vehicle_ids:
            v_track = vehicles_in_frames.filter(pl.col("unique-id") == vehicle_id).sort("frame-count")
            if v_track.height == 0:
                continue

            # Determine vehicle class for this track (safe because v_track is sorted).
            try:
                v_class = int(v_track.get_column("yolo-id")[0])
            except Exception:
                continue

            v1 = v_track.unique(subset=["frame-count"], keep="first")

            # Align person and vehicle by frame-count.
            j = p1.join(v1, on="frame-count", how="inner", suffix="_v")
            shared = j.height
            if shared < min_shared_frames:
                continue

            # Extract aligned arrays.
            p_xy = j.select(["x-center", "y-center"]).to_numpy()
            v_xy = j.select(["x-center_v", "y-center_v"]).to_numpy()

            px = p_xy[:, 0]
            py = p_xy[:, 1]
            vx = v_xy[:, 0]
            vy = v_xy[:, 1]

            p_w = j.get_column("width").to_numpy()
            p_h = j.get_column("height").to_numpy()

            v_w = j.get_column("width_v").to_numpy()
            v_h = j.get_column("height_v").to_numpy()

            # ---------------------------------------------------------------
            # Branch A: OPEN RIDERS (bicycle/motorcycle) => existing logic
            # ---------------------------------------------------------------
            if v_class in open_rider_class_ids:
                dist = np.linalg.norm(p_xy - v_xy, axis=1)
                dist_rel = dist / np.maximum(p_h, eps)
                prox = dist_rel < dist_rel_thresh
                if float(prox.mean()) < prox_req:
                    continue

                relx = vx - px
                rely = vy - py

                spatial = (np.abs(relx) < alpha_x * p_w) & (rely > beta_y * p_h) & (rely < gamma_y * p_h)
                coloc = prox & spatial
                coloc_ratio = float(coloc.mean())

                # Motion similarity (optional fallback).
                p_mov = np.diff(p_xy, axis=0)
                v_mov = np.diff(v_xy, axis=0)

                sim_ratio = 0.0
                if p_mov.shape[0] > 0:
                    na = np.linalg.norm(p_mov, axis=1)
                    nb = np.linalg.norm(v_mov, axis=1)
                    move_mask = (na > eps) & (nb > eps)

                    cos = np.zeros_like(na, dtype=float)
                    cos[move_mask] = (p_mov[move_mask] * v_mov[move_mask]).sum(axis=1) / (na[move_mask] * nb[move_mask])  # noqa: E501

                    prox_steps = prox[1:]
                    m = min(len(prox_steps), len(cos), len(move_mask))
                    prox_steps = prox_steps[:m]
                    cos = cos[:m]
                    move_mask = move_mask[:m]

                    denom_mask = prox_steps & move_mask
                    denom = int(denom_mask.sum())
                    if denom >= min_motion_steps:
                        sim_ratio = float(((cos > sim_thresh) & denom_mask).sum() / denom)

                # Short overlap guard.
                if shared < short_shared_frames:
                    if shared > 1:
                        p_disp = float(np.linalg.norm(p_xy[-1] - p_xy[0]))
                        p_disp_rel = p_disp / float(np.maximum(np.mean(p_h), eps))
                    else:
                        p_disp_rel = 0.0

                    if not (sim_ratio >= short_sim_req or p_disp_rel >= short_disp_req):
                        continue

                ok = (coloc_ratio >= coloc_req) or (sim_ratio >= sim_req and coloc_ratio >= motion_coloc_min)
                if ok:
                    return True

                continue  # open rider branch done

            # ---------------------------------------------------------------
            # Branch B: ENCLOSED VEHICLES (car/bus/truck/train) => logic
            # ---------------------------------------------------------------
            if v_class in enclosed_vehicle_class_ids:
                # 1) Proximity relative to VEHICLE size (scale-free).
                dist = np.linalg.norm(p_xy - v_xy, axis=1)
                dist_rel = dist / np.maximum(v_h, eps)
                prox = dist_rel < enclosed_dist_rel_thresh
                prox_ratio = float(prox.mean())
                if prox_ratio < enclosed_prox_req:
                    continue

                # 2) Person center inside vehicle box (common for driver/passenger detections).
                dx = px - vx
                dy = py - vy
                inside = (np.abs(dx) <= (v_w / 2.0)) & (np.abs(dy) <= (v_h / 2.0))
                inside_ratio = float(inside.mean())
                if inside_ratio < enclosed_inside_req:
                    continue

                # 3) Relative-position stability: passenger/driver stays in roughly same place in vehicle.
                relx_norm = dx / np.maximum(v_w, eps)
                rely_norm = dy / np.maximum(v_h, eps)
                rel_stab = max(robust_range_np(relx_norm), robust_range_np(rely_norm))
                if rel_stab > enclosed_rel_stab_max:
                    continue

                # 4) Optional: low IoU requirement (kept permissive; many person-in-car boxes are tiny).
                # This helps reject odd cases where center is "inside" due to jitter but boxes never overlap.
                iou = iou_xywh_norm(px, py, p_w, p_h, vx, vy, v_w, v_h)
                if float(np.quantile(iou, 0.5)) < enclosed_iou_req:
                    continue

                # 5) Short overlap guard: require stronger inside ratio for very short overlaps.
                if shared < short_shared_frames and inside_ratio < max(enclosed_inside_req, 0.6):
                    continue

                # If all enclosed-vehicle tests pass, treat as vehicle-associated (exclude from crossings).
                return True

        # No vehicle track matched rider/occupant criteria.
        return False

    @staticmethod
    def is_valid_crossing(df, person_id, ratio_thresh=0.6, STATIC_CLASS_IDS=(9, 10, 11, 12, 13),
                          MIN_SHARED_FRAMES=8, RELX_MIN=0.05, Q=0.05, EPS=1e-9):
        """
        Checks whether an apparent pedestrian road-crossing is real or caused by dashcam turning.

        This function is designed for dashcam footage where camera motion (especially turning)
        can create *apparent* lateral motion of pedestrians that are actually stationary.
        To reduce these false positives, it uses detections from "static-ish" objects
        (e.g., traffic lights / stop signs) as a proxy for camera-induced motion.

        Core idea:
          - During a camera turn, both the pedestrian and background objects shift similarly
            in image space, especially in the X direction.
          - If the pedestrian's X motion is mostly explained by the camera (as estimated from
            a static object's X motion), then the pedestrian did not truly move relative to
            the scene and the crossing is likely invalid.

        The algorithm:
          1) Extract the person track (YOLO person class, same unique-id).
          2) Within the person time window, find tracks for static objects.
          3) For each static track, align it with the person by frame-count (inner join).
          4) Compute robust X-motion ranges using quantiles to reduce jitter:
               px_rng   = robust_range(person_x)
               sx_rng   = robust_range(static_x)
               relx_rng = robust_range(person_x - static_x)
             where robust_range(x) = quantile(1-Q) - quantile(Q)
          5) Select the best static reference (most overlap frames; tie-break by larger sx_rng).
          6) Decide validity:
               - If relx_rng is tiny => person moves with camera => invalid (False)
               - Else if sx_rng/px_rng is large AND relx_rng not strong => invalid (False)
               - Otherwise => valid (True)

        Notes:
          - This function assumes the input coordinates are normalised in [0, 1].
          - It expects `df` to be a Polars DataFrame and `pl` to be imported.
          - If no static objects are available, the function returns True (cannot verify).

        Args:
          df (pl.DataFrame): YOLO detections with columns:
            - "yolo-id" (int): class id (0 = person)
            - "unique-id": tracker id per object
            - "frame-count" (int): frame index
            - "x-center" (float): normalised x-center in [0,1]
            - "confidence" (float, optional): detection confidence
            - other YOLO fields are allowed but not required here
          person_id (Any): The tracker unique-id for the person to validate.
          ratio_thresh (float): Threshold for camera-dominance ratio = static_x_rng / person_x_rng.
            Larger values are more permissive. Typical range: 0.5–0.9.
          STATIC_CLASS_IDS (Tuple[int, ...]): Class IDs treated as static references.
            Default is COCO-like: traffic light (9), fire hydrant (10), stop sign (11),
            parking meter (12), bench (13).
          MIN_SHARED_FRAMES (int): Minimum number of overlapping frames between person and a
            candidate static track to consider it usable.
          RELX_MIN (float): Minimum robust range of (person_x - static_x) to treat motion as
            real (independent of camera). Lower = more permissive.
          Q (float): Quantile used for robust range (e.g., Q=0.05 uses 5%..95% range).
          EPS (float): Small constant to avoid divide-by-zero.

        Returns:
          bool: True if the crossing is likely valid (person moved independently of the camera),
            False if the apparent crossing is likely caused by camera turning.

        """
        # -------------------------------------------------------------------------
        # Deduplicate per frame to reduce jitter and avoid join misalignment.
        #    - For each (yolo-id, unique-id, frame-count), keep the highest-confidence row.
        #    - This is important because multiple detections per frame can inflate ranges.
        # -------------------------------------------------------------------------
        if "confidence" in df.columns:
            df = (
                df.sort(
                    ["yolo-id", "unique-id", "frame-count", "confidence"],
                    descending=[False, False, False, True],
                )
                .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
            )
        else:
            # If confidence is not present, just keep the first per key.
            df = df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

        # -------------------------------------------------------------------------
        # Extract the person's track.
        #    - We restrict to yolo-id == 0 (person) to avoid accidental collisions where
        #      another class might share the same unique-id due to tracker re-use.
        #    - We also ensure one row per frame-count for a clean alignment later.
        # -------------------------------------------------------------------------
        person_track = (
            df.filter((pl.col("yolo-id") == 0) & (pl.col("unique-id") == person_id))
            .sort("frame-count")
            .unique(subset=["frame-count"], keep="first")
        )
        if person_track.height == 0:
            # No track => cannot validate => treat as invalid crossing.
            return False

        # Identify the time window of the person track.
        frames = person_track.get_column("frame-count").to_numpy()
        first_frame = int(frames.min())
        last_frame = int(frames.max())

        # -------------------------------------------------------------------------
        # Collect static-object detections in the same time window.
        #    - These objects should (ideally) be fixed in the world and move only due to
        #      camera motion. Their apparent X movement serves as a proxy for camera turn.
        # -------------------------------------------------------------------------
        static_objs = (
            df.filter(
                (pl.col("frame-count") >= first_frame)
                & (pl.col("frame-count") <= last_frame)
                & (pl.col("yolo-id").is_in(list(STATIC_CLASS_IDS)))
            )
            .sort("frame-count")
        )

        # If we have no static references, we cannot disentangle camera motion.
        # Preserve our earlier behavior: assume the crossing is valid.
        if static_objs.height == 0:
            return True

        # -------------------------------------------------------------------------
        # Define a robust range function.
        #    - Using min/max can be overly sensitive to jitter/outliers.
        #    - Quantile range (Q..1-Q) is more stable in practice.
        # -------------------------------------------------------------------------
        def robust_range(series: pl.Series) -> float:
            # Quantiles might fail if the series is empty or not numeric; handle gracefully.
            try:
                lo = series.quantile(Q, "nearest")
                hi = series.quantile(1.0 - Q, "nearest")
                return float(hi - lo)  # type: ignore
            except Exception:
                return 0.0

        # -------------------------------------------------------------------------
        # Compare the person to each static track and choose the best reference.
        #    Selection policy:
        #      - prefer the static track with the most overlapping frames
        #      - tie-break by larger static motion (more informative camera signal)
        # -------------------------------------------------------------------------
        best = None
        static_uids = static_objs.select("unique-id").unique().to_series().to_list()

        for sid in static_uids:
            # Extract a single static object's track and keep one row per frame.
            s_track = (
                static_objs.filter(pl.col("unique-id") == sid)
                .sort("frame-count")
                .unique(subset=["frame-count"], keep="first")
            )
            if s_track.height == 0:
                continue

            # Align by frame-count. We only consider frames where both are present.
            joined = person_track.join(s_track, on="frame-count", how="inner", suffix="_s")

            # Require a minimum overlap to avoid unstable statistics.
            if joined.height < MIN_SHARED_FRAMES:
                continue

            # Extract aligned X-centers.
            px = joined.get_column("x-center")      # person x
            sx = joined.get_column("x-center_s")    # static x
            relx = px - sx                          # person relative to static

            # Robust motion magnitudes.
            px_rng = robust_range(px)
            sx_rng = robust_range(sx)
            relx_rng = robust_range(relx)

            # Camera dominance ratio: if high, camera motion can explain most person motion.
            ratio = float(sx_rng / max(px_rng, EPS))

            cand = {
                "shared": int(joined.height),
                "px_rng": float(px_rng),
                "sx_rng": float(sx_rng),
                "relx_rng": float(relx_rng),
                "ratio": float(ratio),
            }

            # Pick best candidate reference.
            if best is None:
                best = cand
            else:
                if (cand["shared"], cand["sx_rng"]) > (best["shared"], best["sx_rng"]):
                    best = cand

        # If no static track had enough overlap, fall back to permissive behavior.
        if best is None:
            return True

        # -------------------------------------------------------------------------
        # Decision rules (updated to prioritise RELATIVE motion).
        #
        # Rule A (primary):
        #   If the relative motion is tiny, the person is moving with the background
        #   and the "crossing" is likely a camera-turn artifact => invalid.
        #
        # Rule B (secondary guard):
        #   If camera dominance ratio is high *and* relative motion is not strong,
        #   treat as invalid.
        #
        # These rules intentionally rely on (person_x - static_x), which is the key
        # to rejecting turning artifacts.
        # -------------------------------------------------------------------------
        if best["relx_rng"] < RELX_MIN:
            return False

        if best["ratio"] >= float(ratio_thresh) and best["relx_rng"] < (2.0 * RELX_MIN):
            return False

        # Otherwise, the person shows independent lateral motion relative to the scene.
        return True
