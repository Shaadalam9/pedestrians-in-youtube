import numpy as np
import polars as pl
from utils.core.metadata import MetaData

metadata_class = MetaData()


class Detection:

    def __init__(self) -> None:
        pass

    def pedestrian_crossing(self, dataframe, video_id, df_mapping, min_x, max_x, person_id):
        """Counts the number of person with a specific ID crosses the road within specified boundaries.

        Args:
            dataframe (DataFrame): DataFrame containing data from the video.
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
        crossed_ids = (
            crossed_df
            .group_by("unique-id")
            .agg([
                pl.col("x-center").min().alias("_x_min"),
                pl.col("x-center").max().alias("_x_max"),
            ])
            .filter((pl.col("_x_min") <= min_x) & (pl.col("_x_max") >= max_x))
            .select("unique-id")
            .to_series()
            .to_list()
        )

        # Lookup avg_height via existing MetaData helper (convert mapping once if needed)
        avg_height = None
        # df_mapping_for_meta = df_mapping.to_pandas() if isinstance(df_mapping, pl.DataFrame) else df_mapping
        result = metadata_class.find_values_with_video_id(df_mapping, video_id)
        if result is not None:
            avg_height = result[15]

        pedestrian_ids = []
        for uid in crossed_ids:
            if self.is_rider_id(dataframe, uid, avg_height):
                continue  # Filter out riders
            if not self.is_valid_crossing(dataframe, uid):
                continue  # Skip fake crossing
            pedestrian_ids.append(uid)

        return pedestrian_ids, crossed_ids

    def is_rider_id(self, df, id, avg_height, min_shared_frames=5,
                    dist_thresh=80, similarity_thresh=0.8, overlap_ratio=0.7):
        """
        Determines if a person identified by the given unique-id is riding a bicycle or motorcycle
        during their trajectory in the YOLO detection DataFrame.

        The function checks, for the duration in which the person is present, whether a bicycle or
        motorcycle detection is present and moving together (i.e., close proximity and similar
        movement direction and speed) with the person for a sufficient number of frames. If so,
        the person is likely a cyclist or motorcyclist and should be excluded from pedestrian analysis.

        Args:
            df (pd.DataFrame): YOLO detections DataFrame containing columns:
                'yolo-id' (class, 0=person, 1=bicycle, 3=motorcycle), 'unique-id',
                'frame-count', 'x-center', 'y-center', 'width', 'height'.
            id (int or str): The unique-id of the person to analyse.
            avg_height (float): The average real-world height of the person (cm).
            min_shared_frames (int, optional): Minimum number of frames with both the person and vehicle
                present for comparison. Defaults to 5.
            dist_thresh (float, optional): Maximum distance (in pixels) between person and vehicle
                bounding box centers to be considered "moving together". Defaults to 50.
            similarity_thresh (float, optional): Minimum cosine similarity threshold for movement direction
                to be considered "similar". Ranges from -1 (opposite) to 1 (identical direction). Defaults to 0.8.
            overlap_ratio (float, optional): Fraction of overlapping frames where proximity and movement
                similarity must be satisfied. Defaults to 0.7 (i.e., 70%).

        Returns:
            bool: True if the person is moving together with a bicycle or motorcycle (i.e., is a rider);
                False if likely a pedestrian.

        Example:
            for id, time in id_time.items():
                if is_rider_id(df, id):
                    continue  # Skip this id (cyclist or motorcyclist)
                # ...process as pedestrian...

        """
        # If we cannot compute pixels_per_cm, do not classify as rider
        try:
            if avg_height is None or float(avg_height) == 0.0:
                return False
            avg_height_f = float(avg_height)
        except Exception:
            return False

        person_track = df.filter(pl.col("unique-id") == id)
        if person_track.height == 0:
            return False

        frames = person_track.get_column("frame-count").to_numpy()
        if frames.size < min_shared_frames:
            return False

        first_frame = int(frames.min())
        last_frame = int(frames.max())

        vehicles_in_frames = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in([1, 3]))
        )

        if vehicles_in_frames.height == 0:
            return False

        vehicle_ids = vehicles_in_frames.select("unique-id").unique().to_series().to_list()

        person_frames_all = person_track.get_column("frame-count").to_numpy()

        for vehicle_id in vehicle_ids:
            vehicle_track = vehicles_in_frames.filter(pl.col("unique-id") == vehicle_id)
            if vehicle_track.height == 0:
                continue

            vehicle_frames_all = vehicle_track.get_column("frame-count").to_numpy()

            shared_frames = np.intersect1d(person_frames_all, vehicle_frames_all)
            if shared_frames.size < min_shared_frames:
                continue

            # Align positions on shared frames (sorted by frame-count)
            person_shared = (
                person_track
                .filter(pl.col("frame-count").is_in(shared_frames))
                .sort("frame-count")
            )
            vehicle_shared = (
                vehicle_track
                .filter(pl.col("frame-count").is_in(shared_frames))
                .sort("frame-count")
            )

            # Safety: if counts diverge due to duplicates, align by truncation
            n = min(person_shared.height, vehicle_shared.height)
            if n < min_shared_frames:
                continue

            person_pos = person_shared.select(["x-center", "y-center"]).head(n).to_numpy()
            vehicle_pos = vehicle_shared.select(["x-center", "y-center"]).head(n).to_numpy()

            person_heights = person_shared.get_column("height").head(n).to_numpy()
            # pixels per cm per frame
            pixels_per_cm = person_heights / avg_height_f

            # avoid division by zero
            if np.any(pixels_per_cm == 0):
                continue

            pixel_dists = np.linalg.norm(person_pos - vehicle_pos, axis=1)
            distances_cm = pixel_dists / pixels_per_cm

            proximity = distances_cm < dist_thresh
            if proximity.sum() / len(distances_cm) < overlap_ratio:
                continue

            person_mov = np.diff(person_pos, axis=0)
            vehicle_mov = np.diff(vehicle_pos, axis=0)

            similarities = []
            for a, b in zip(person_mov, vehicle_mov):
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na == 0 or nb == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(float(np.dot(a, b) / (na * nb)))

            if len(similarities) == 0:
                continue

            similarity_mask = np.array(similarities) > similarity_thresh
            if similarity_mask.sum() / len(similarities) >= overlap_ratio:
                return True

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
