from tqdm import tqdm
import math
from collections import defaultdict
from utils.core.metadata import MetaData

metadata_class = MetaData()


class Grouping:
    def __init__(self) -> None:
        pass

    def city_country_wrapper(self, input_dict, mapping, show_progress=False):
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
        # Choose the right iterator based on flag
        iterator = input_dict.items()

        if show_progress:
            iterator = tqdm(iterator, desc="Wrapping by city/condition", total=len(input_dict))

        for key, value in iterator:
            # Lookup metadata associated with the video ID
            result = metadata_class.find_values_with_video_id(mapping, key)

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
        state = metadata_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "state")

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

    def country_averages_from_nested(self, var_dict, df_mapping):
        """Aggregates nested city-level values into country-level averages by condition.

        This method processes a dictionary where keys are formatted as
        `"{city}_{lat}_{long}_{condition}"` and values are dictionaries of measurements.
        It groups all measurements by `country_condition` (e.g., `"USA_0"`, `"USA_1"`)
        and computes the average for each group.

        Args:
            var_dict (dict): Dictionary of city-condition data, where:
                - Keys (str): "{city}_{lat}_{long}_{condition}".
                - Values (dict): Inner dictionary of measurements for that city-condition.
            df_mapping (pd.DataFrame): Mapping DataFrame containing city-to-country info.

        Returns:
            dict: A dictionary mapping "country_condition" → average value (float).

        Example:
            >>> var_dict = {
            ...     "Paris_48.85_2.35_0": {"a": 10, "b": 20},
            ...     "Paris_48.85_2.35_1": {"a": 30, "b": 50}
            ... }
            >>> df_mapping
               city     lat  country
            0  Paris  48.85  France
            >>> country_averages_from_nested(var_dict, df_mapping)
            {'France_0': 15.0, 'France_1': 40.0}
        """
        # Store all values grouped by country and condition (e.g., "France_0")
        country_condition_values = defaultdict(list)

        # Iterate over each city-condition entry
        for k, inner_dict in var_dict.items():
            # Split key into components: city, latitude, longitude, condition
            city, lat, long, condition = k.rsplit('_', 3)
            lat = float(lat)
            condition = int(condition)

            # Map city+lat to country using df_mapping
            country = metadata_class.get_value(df_mapping, "city", city, "lat", lat, "country")
            if country:
                key = f"{country}_{condition}"
                # Append all measurement values from the inner dictionary
                country_condition_values[key].extend(inner_dict.values())

        # Compute average for each country-condition if values exist
        country_condition_averages = {
            key: sum(vals) / len(vals) for key, vals in country_condition_values.items() if vals
        }

        return country_condition_averages

    def country_averages_from_flat(self, var_dict, df_mapping):
        """Aggregates flat city-level values into country-level averages by condition.

        This method processes a dictionary where keys are formatted as
        `"{city}_{lat}_{long}_{condition}"` and values are numeric measurements.
        It groups all measurements by `country_condition` (e.g., `"USA_0"`, `"USA_1"`)
        and computes the average for each group.

        Args:
            var_dict (dict): Dictionary of city-condition values, where:
                - Keys (str): "{city}_{lat}_{long}_{condition}".
                - Values (float or int): Numeric measurement for that city-condition.
            df_mapping (pd.DataFrame): Mapping DataFrame containing city-to-country info.

        Returns:
            dict: A dictionary mapping "country_condition" → average value (float).

        Example:
            >>> var_dict = {
            ...     "Paris_48.85_2.35_0": 10,
            ...     "Paris_48.85_2.35_1": 30
            ... }
            >>> df_mapping
               city     lat  country
            0  Paris  48.85  France
            >>> country_averages_from_flat(var_dict, df_mapping)
            {'France_0': 10.0, 'France_1': 30.0}
        """
        # Store all values grouped by country and condition (e.g., "France_0")
        country_condition_values = defaultdict(list)

        # Iterate over each city-condition entry
        for k, v in var_dict.items():
            # Split key into components: city, latitude, longitude, condition
            city, lat, long, condition = k.rsplit('_', 3)
            lat = float(lat)
            condition = int(condition)

            # Map city+lat to country using df_mapping
            country = metadata_class.get_value(df_mapping, "city", city, "lat", lat, "country")
            if country:
                key = f"{country}_{condition}"
                # Append the numeric measurement
                country_condition_values[key].append(v)

        # Compute average for each country-condition
        country_condition_averages = {
            key: sum(vals) / len(vals) for key, vals in country_condition_values.items()
        }

        return country_condition_averages

    def country_sum_from_cities(self, var_dict, df_mapping):
        """Aggregates city-level numeric values into country-level sums by condition.

        This method processes a dictionary where keys are formatted as
        `"{city}_{lat}_{long}_{condition}"` and values are numeric measurements.
        It groups all measurements by `country_condition` (e.g., `"USA_0"`, `"USA_1"`)
        and computes the sum for each group.

        Args:
            var_dict (dict): Dictionary of city-condition values, where:
                - Keys (str): "{city}_{lat}_{long}_{condition}".
                - Values (float or int): Numeric measurement for that city-condition.
            df_mapping (pd.DataFrame): Mapping DataFrame containing city-to-country info.

        Returns:
            dict: A dictionary mapping "country_condition" → summed value (float or int).

        Example:
            >>> var_dict = {
            ...     "Paris_48.85_2.35_0": 5,
            ...     "Paris_48.85_2.35_1": 7
            ... }
            >>> df_mapping
               city     lat  country
            0  Paris  48.85  France
            >>> country_sum_from_cities(var_dict, df_mapping)
            {'France_0': 5, 'France_1': 7}
        """
        # Store all values grouped by country and condition (e.g., "France_0")
        country_condition_values = defaultdict(list)

        # Iterate over each city-condition entry
        for k, v in var_dict.items():
            # Split key into components: city, latitude, longitude, condition
            city, lat, long, condition = k.rsplit('_', 3)
            lat = float(lat)
            condition = int(condition)

            # Map city+lat to country using df_mapping
            country = metadata_class.get_value(df_mapping, "city", city, "lat", lat, "country")
            if country:
                key = f"{country}_{condition}"
                # Append the numeric measurement
                country_condition_values[key].append(v)

        # Compute sum for each country-condition
        country_condition_sums = {
            key: sum(vals) for key, vals in country_condition_values.items()
        }

        return country_condition_sums
