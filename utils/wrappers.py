import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
from .values import Values
import pycountry

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
                (_, start, end, condition, city, state, country, gdp_, population, population_country,
                 traffic_mortality_, continent, literacy_rate, avg_height, iso3, fps) = result

                # Create the grouping key
                grouping_key = f"{city}_{state}_{condition}"

                # Initialize the dictionary for the grouping key if it doesn't exist
                if grouping_key not in output:
                    output[grouping_key] = {}

                # Add or update the entry for this video ID
                output[grouping_key][key] = value

        return output

    def format_city_state(self, city_state):
        """
        Formats a city_state string or a list of strings in the format 'City_State'.
        If the state is 'unknown', only the city is returned.
        Handles cases where the format is incorrect or missing the '_'.

        Args:
            city_state (str or list): A single string or list of strings in the format 'City_State'.

        Returns:
            str or list: A formatted string or list of formatted strings in the format 'City, State' or 'City'.
        """
        if isinstance(city_state, str):  # If input is a single string
            if "_" in city_state:
                city, state = city_state.split("_", 1)
                return f"{city}, {state}" if state.lower() != "unknown" else city
            else:
                return city_state  # Return as-is if no '_' in string
        elif isinstance(city_state, list):  # If input is a list
            formatted_list = []
            for cs in city_state:
                if "_" in cs:
                    city, state = cs.split("_", 1)
                    if state.lower() != "unknown":
                        formatted_list.append(f"{city}, {state}")
                    else:
                        formatted_list.append(city)
                else:
                    formatted_list.append(cs)  # Append as-is if no '_'
            return formatted_list
        else:
            raise TypeError("Input must be a string or a list of strings.")

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
