import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
from .values import Values
import pycountry
import math

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

values_class = Values()


class Wrappers():
    def __init__(self) -> None:
        pass

    def city_country_wrapper(self, input_dict, mapping):
        """
        Processes an input dictionary of video IDs and their corresponding values, maps each video ID to metadata
        using a provided mapping function, and builds an output dictionary keyed by city and condition.
        Wrapes in the form of {city_condition: {video-id_start-time: {unique-id: parameter}}}

        Args:
            input_dict (dict): A dictionary where keys are video IDs and values are associated data.
            mapping: A mapping resource or structure (currently unused in this function,
                 possibly used internally by Analysis.find_values_with_video_id).

        Returns:
            dict: A dictionary where each key is a string formatted as "City_Condition" and
                  each value is a dictionary of the original key-value pair from input_dict.
        """

        output = {}
        for key, value in input_dict.items():
            # Lookup metadata associated with the video ID
            result = values_class.find_values_with_video_id(mapping, key)

            # Check if the result is None (i.e., no matching data was found)
            if result is not None:
                # Unpack the returned metadata
                condition = result[3]
                city = result[4]
                lat = result[6]
                long = result[7]

                # Create the grouping key
                grouping_key = f"{city}_{lat}_{long}_{condition}"

                # Initialise the dictionary for the grouping key if it doesn't exist
                if grouping_key not in output:
                    output[grouping_key] = {}

                # Add or update the entry for this video ID
                output[grouping_key][key] = value

        return output

    def process_city_string(self, city, df_mapping):
        """
        Splits a city string into its components, retrieves the state using provided helper classes,
        and formats the city string with its state.

        Args:
            city (str): The city string in the format "city_lat_long", e.g., "Chicago_41.8781_-87.6298".
            df_mapping: The DataFrame or mapping object required by values_class.get_value.
            values_class: An object/class with a get_value method to retrieve data such as state.
            wrapper_class: An object/class with a format_city_lat_lon method to format the city string.

        Returns:
            str: The formatted city string with its state, as returned by wrapper_class.format_city_lat_lon.
        """
        # Split the city string into city name, latitude, and longitude
        city_new, lat, long = city.split('_')

        # Use values_class to retrieve the state associated with the city and coordinates
        state = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "state")

        # Use wrapper_class to format the city string with state (ignoring type checking if needed)
        formatted_city = self.format_city_lat_lon(city, state)  # type: ignore

        return formatted_city

    def format_city_lat_lon(self, city_lat_lon, state):
        """
        Formats the city from a city_latitude_longitude string or list,
        returning '{city}, {state}' if state is provided and not nan, else just '{city}'.

        Args:
            city_lat_lon (str or list): String or list of 'City_Latitude_Longitude'.
            state (str or list): Corresponding state(s), can be 'nan'.

        Returns:
            str or list: Formatted string(s).
        """

        def is_nan(value):
            # Handles both float('nan') and string 'nan'
            missing = {'nan', 'na', 'n/a', ''}
            return (
                value is None
                or (isinstance(value, float) and math.isnan(value))
                or (isinstance(value, str) and value.strip().lower() in missing)
            )

        def format_single(city_entry, state_entry):
            city = city_entry.split("_")[0] if "_" in city_entry else city_entry
            if not is_nan(state_entry):
                return f"{city}, {state_entry}"
            else:
                return city

        if isinstance(city_lat_lon, str):
            # Expecting state as str as well
            return format_single(city_lat_lon, state)

        elif isinstance(city_lat_lon, list):
            # Expecting state as list
            if not isinstance(state, list) or len(city_lat_lon) != len(state):
                raise ValueError("city_lat_lon and state must both be lists of the same length.")
            return [format_single(c, s) for c, s in zip(city_lat_lon, state)]

        else:
            raise TypeError("city_lat_lon must be a string or a list of strings.")

    def iso2_to_flag(self, iso2):
        """
        Convert a 2-letter ISO country code to a corresponding emoji flag.

        This uses Unicode regional indicator symbols to represent flags. For example, 'US' becomes 🇺🇸.

        Args:
            iso2 (str or None): Two-letter ISO country code (e.g., 'US'). If None, returns a placeholder.

        Returns:
            str: A flag emoji corresponding to the country code, or 🇽🇰 (Kosovo) as a fallback.
        """
        logger.debug(f"Converting iso2 {iso2} to flag.")
        if iso2 is None:
            # Return a placeholder or an empty string if the ISO-2 code is not available
            logger.debug("Set ISO-2 to Kosovo.")
            return "🇽🇰"
        return chr(ord('🇦') + (ord(iso2[0]) - ord('A'))) + chr(ord('🇦') + (ord(iso2[1]) - ord('A')))

    def iso3_to_iso2(self, iso3_code):
        """
        Convert an ISO 3166-1 alpha-3 country code (e.g., 'USA') to an ISO 3166-1 alpha-2 code (e.g., 'US').

        Args:
            iso3_code (str): The three-letter country code (ISO alpha-3).

        Returns:
            str or None: The corresponding two-letter country code (ISO alpha-2),
                         or None if the code is invalid or not found.
        """
        try:
            # Look up the country using pycountry
            country = pycountry.countries.get(alpha_3=iso3_code)
            return country.alpha_2 if country else None
        except (AttributeError, LookupError) as e:
            logger.debug(f"Converting up ISO-3 {iso3_code} to ISO-2 returned error: {e}.")
            return None
