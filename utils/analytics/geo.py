import re
import ast
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger


class Geo:

    def __init__(self) -> None:
        pass

    @staticmethod
    def find_city_id(df, video_id, start_time):
        """
        Find the city identifier (row 'id') associated with a given video ID and start time.

        This function iterates through a DataFrame where each row may reference multiple videos and
        corresponding start times (stored as lists). It returns the 'id' value from the row where both
        the video ID and the exact start time match.

        Args:
            df (pd.DataFrame): DataFrame containing at least:
                               - 'videos': a string representing a list of video IDs.
                               - 'start_time': a string representing a list of lists of start times.
                               - 'id': unique identifier for each row (e.g., city or condition group).
            video_id (str): The video filename or identifier to search for.
            start_time (float or int): The specific start time to match within the corresponding list.

        Returns:
            The value of the 'id' field in the matching row, or None if no match is found.
        """
        logger.debug(f"{video_id}: looking for city, start_time={start_time}.")

        # Iterate rows (Polars)
        for row in df.select(["id", "videos", "start_time"]).iter_rows(named=True):
            try:
                videos_raw = row.get("videos")
                start_raw = row.get("start_time")

                if not isinstance(videos_raw, str) or not isinstance(start_raw, str):
                    continue

                # Extract video ids robustly
                videos = re.findall(r"[\w-]+", videos_raw)

                if video_id not in videos:
                    continue

                idx = videos.index(video_id)

                # Parse nested start times
                start_times = ast.literal_eval(start_raw)
                if not isinstance(start_times, list) or idx >= len(start_times):
                    continue

                sub = start_times[idx]
                if not isinstance(sub, list):
                    continue

                # Match start_time (handle int/float/string mismatches)
                try:
                    st_val = float(start_time)
                    sub_vals = []
                    for x in sub:
                        try:
                            sub_vals.append(float(x))
                        except Exception:
                            continue
                    if st_val in sub_vals:
                        return row["id"]
                except Exception:
                    # Fallback: direct membership check
                    if start_time in sub:
                        return row["id"]

            except Exception:
                continue

        return None

    @staticmethod
    def get_coordinates(city, state, country):
        """
        Retrieve geographic coordinates (latitude and longitude) for a given city, state, and country.

        The function uses the Nominatim geocoding service from the geopy library. It first constructs a location
        query string based on the provided city, optional state, and country. A unique user-agent is generated
        for each call to avoid getting blocked by the server.

        Args:
            city (str): Name of the city.
            state (str or None): Name of the state or province. Optional; can be None or 'nan'.
            country (str): Name of the country.

        Returns:
            tuple: A tuple (latitude, longitude) if geocoding is successful, otherwise (None, None).

        Exceptions:
            - Logs and handles `GeocoderTimedOut` if the geocoding request times out.
            - Logs and handles `GeocoderUnavailable` if the geocoding server is unreachable.
        """
        # Generate a unique user agent with the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        user_agent = f"my_geocoding_script_{current_time}"

        # Initialise the Nominatim geocoder with the unique user agent
        geolocator = Nominatim(user_agent=user_agent)

        try:
            # Form the query string depending on whether a valid state value is provided
            if state and str(state).lower() != 'nan':
                location_query = f"{city}, {state}, {country}"  # Combine city, state and country
            else:
                location_query = f"{city}, {country}"  # Combine city and country
            location = geolocator.geocode(location_query, timeout=2)  # type: ignore # Set a 2-second timeout

            if location:
                return location.latitude, location.longitude  # type: ignore
            else:
                logger.error(f"Failed to geocode {location_query}")
                return None, None  # Return None if city is not found

        except GeocoderTimedOut:
            # Handle timeout errors when the request takes too long
            logger.error(f"Geocoding timed out for {location_query}.")
        except GeocoderUnavailable:
            # Handle cases where the geocoding service is not available
            logger.error(f"Geocoding server could not be reached for {location_query}.")
            return None, None  # Return None if city is not found
