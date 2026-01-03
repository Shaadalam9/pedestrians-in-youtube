import numpy as np
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
        crossed_ids = dataframe[(dataframe["yolo-id"] == person_id)]

        # Group entries by Unique ID
        crossed_ids_grouped = crossed_ids.groupby("unique-id")

        # Filter entries based on x-coordinate boundaries
        filtered_crossed_ids = crossed_ids_grouped.filter(
            lambda x: (x["x-center"] <= min_x).any() and (x["x-center"] >= max_x).any())

        # Get unique IDs of the person who crossed the road within boundaries
        crossed_ids = filtered_crossed_ids["unique-id"].unique()

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
        # Extract all rows corresponding to the person id
        person_track = df[df['unique-id'] == id]
        if person_track.empty:
            return False  # No data for this id

        frames = person_track['frame-count'].values
        if len(frames) < min_shared_frames:
            return False  # Not enough frames to perform check

        first_frame, last_frame = frames.min(), frames.max()

        # Filter DataFrame to get all bicycle/motorcycle detections in relevant frames
        mask = (
            (df['frame-count'] >= first_frame)
            & (df['frame-count'] <= last_frame)
            & (df['yolo-id'].isin([1, 3]))
        )
        vehicles_in_frames = df[mask]

        for vehicle_id in vehicles_in_frames['unique-id'].unique():
            # Get trajectory for this vehicle
            vehicle_track = vehicles_in_frames[vehicles_in_frames['unique-id'] == vehicle_id]

            # Find shared frames between person and vehicle
            shared_frames = np.intersect1d(person_track['frame-count'], vehicle_track['frame-count'])

            if len(shared_frames) < min_shared_frames:
                continue  # Not enough overlapping frames to check movement together

            # Align positions for person and vehicle on shared frames, sorted by Frame Count
            person_pos = (
                person_track[person_track['frame-count'].isin(shared_frames)]
                .sort_values('frame-count')[['x-center', 'y-center']].values
            )
            vehicle_pos = (
                vehicle_track[vehicle_track['frame-count'].isin(shared_frames)]
                .sort_values('frame-count')[['x-center', 'y-center']].values
            )

            # Calculate person's bounding box heights in pixels for shared frames
            person_heights = (
                person_track[person_track['frame-count'].isin(shared_frames)]
                .sort_values('frame-count')['height'].values
            )  # This is in pixels per frame

            # Compute pixels-per-cm for each frame using the average real-world height
            pixels_per_cm = person_heights / avg_height  # array, one per frame

            # Compute Euclidean distances between person and vehicle centers for shared frames
            pixel_dists = np.linalg.norm(person_pos - vehicle_pos, axis=1)
            distances_cm = pixel_dists / pixels_per_cm  # Real-world distance per frame

            proximity = (distances_cm < dist_thresh)
            if proximity.sum() / len(distances_cm) < overlap_ratio:
                continue  # Not close enough for sufficient frames

            # Calculate movement vectors (delta positions) for both tracks
            person_mov = np.diff(person_pos, axis=0)
            vehicle_mov = np.diff(vehicle_pos, axis=0)

            # Compute cosine similarity of movement direction for each step
            similarities = []
            for a, b in zip(person_mov, vehicle_mov):
                if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                    similarities.append(0)
                else:
                    similarities.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            similarity_mask = (np.array(similarities) > similarity_thresh)

            if similarity_mask.sum() / len(similarities) >= overlap_ratio:
                return True

        # If no such vehicle found moving together, label as pedestrian
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

        # Extract all detections for the specified person
        person_track = df[df['unique-id'] == person_id]
        if person_track.empty:
            # No detection for this person
            return False

        # Determine the first and last frames in which the person appears
        frames = person_track['frame-count'].values
        first_frame, last_frame = frames.min(), frames.max()

        # Filter for static objects (traffic light or stop sign) in those frames
        static_objs = df[
            (df['frame-count'] >= first_frame) &
            (df['frame-count'] <= last_frame) &
            (df['yolo-id'].isin([9, 11]))
        ]

        if static_objs.empty:
            # No static object present during this period; cannot verify, assume valid
            return True

        # Calculate the y-movement (vertical movement) of the person
        person_y_movement = person_track['y-center'].max() - person_track['y-center'].min()

        # Calculate the maximum y-movement among all static objects in the same frame range
        max_static_y_movement = 0
        for obj_id in static_objs['unique-id'].unique():
            obj_track = static_objs[static_objs['unique-id'] == obj_id]
            y_movement = obj_track['y-center'].max() - obj_track['y-center'].min()
            if y_movement > max_static_y_movement:
                max_static_y_movement = y_movement

        # If the person does not move vertically, it's an invalid crossing
        if person_y_movement == 0:
            return False

        # Compute the movement ratio
        movement_ratio = max_static_y_movement / person_y_movement

        # If the static object moves too much compared to the person, crossing is invalid
        if movement_ratio > ratio_thresh:
            return False
        else:
            return True
