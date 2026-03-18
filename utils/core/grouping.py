from tqdm import tqdm
import math
from collections import defaultdict
import polars as pl

from utils.core.metadata import MetaData

metadata_class = MetaData()


class Grouping:
    def __init__(self) -> None:
        pass

    def locality_country_wrapper(self, input_dict, mapping: pl.DataFrame, show_progress: bool = False):
        """
        Processes an input dictionary of video IDs and their corresponding values, maps each video ID to metadata
        using a provided mapping function, and builds an output dictionary keyed by locality and condition.
        Wrapes in the form of {locality_condition: {video-id_start-time: {unique-id: parameter}}}

        Args:
            input_dict (dict): A dictionary where keys are video IDs and values are associated data.
            mapping: A mapping resource or structure (currently unused in this function,
                 possibly used internally by Analysis.find_values_with_video_id).

        Returns:
            dict: A dictionary where each key is a string formatted as "locality_condition" and
                  each value is a dictionary of the original key-value pair from input_dict.
        """

        output = {}

        iterator = input_dict.items()
        if show_progress:
            iterator = tqdm(iterator, desc="Wrapping by locality/condition", total=len(input_dict))

        for key, value in iterator:
            result = metadata_class.find_values_with_video_id(mapping, key)
            if result is None:
                continue

            condition = result[3]
            locality = result[4]
            lat = result[6]
            long = result[7]

            grouping_key = f"{locality}_{lat}_{long}_{condition}"
            if grouping_key not in output:
                output[grouping_key] = {}

            output[grouping_key][key] = value

        return output

    def process_locality_string(self, locality: str, df_mapping: pl.DataFrame):
        """
        Splits a locality string into its components, retrieves the state using provided helper classes,
        and formats the locality string with its state.

        Args:
            locality (str): The locality string in the format "locality_lat_long", e.g., "Chicago_41.8781_-87.6298".
            df_mapping: The DataFrame or mapping object required by values_class.get_value.
            values_class: An object/class with a get_value method to retrieve data such as state.
            wrapper_class: An object/class with a format_locality_lat_lon method to format the locality string.

        Returns:
            str: The formatted locality string with its state, as returned by wrapper_class.format_locality_lat_lon.
        """
        # Split the locality string into locality name, latitude, and longitude
        locality_new, lat, long = locality.split("_")

        # Use values_class to retrieve the state associated with the locality and coordinates
        state = metadata_class.get_value(df_mapping, "locality", locality_new, "lat", float(lat), "state")

        # Use wrapper_class to format the locality string with state (ignoring type checking if needed)
        formatted_locality = self.format_locality_lat_lon(locality, state)  # type: ignore

        return formatted_locality

    def format_locality_lat_lon(self, locality_lat_lon, state):
        """
        Formats the locality from a locality_latitude_longitude string or list,
        returning '{locality}, {state}' if state is provided and not nan, else just '{locality}'.

        Args:
            locality_lat_lon (str or list): String or list of 'locality_Latitude_Longitude'.
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

        def format_single(locality_entry, state_entry):
            locality = locality_entry.split("_")[0] if "_" in locality_entry else locality_entry
            if not is_nan(state_entry):
                return f"{locality}, {state_entry}"
            else:
                return locality

        if isinstance(locality_lat_lon, str):
            # Expecting state as str as well
            return format_single(locality_lat_lon, state)

        elif isinstance(locality_lat_lon, list):
            # Expecting state as list
            if not isinstance(state, list) or len(locality_lat_lon) != len(state):
                raise ValueError("locality_lat_lon and state must both be lists of the same length.")
            return [format_single(c, s) for c, s in zip(locality_lat_lon, state)]

        else:
            raise TypeError("locality_lat_lon must be a string or a list of strings.")

    def format_locality_state(self, locality_state):
        """
        Formats a locality_state string or a list of strings in the format 'locality_State'.
        If the state is 'unknown', only the locality is returned.
        Handles cases where the format is incorrect or missing the '_'.

        Args:
            locality_state (str or list): A single string or list of strings in the format 'locality_State'.

        Returns:
            str or list: A formatted string or list of formatted strings in the format 'locality, State' or 'locality'.
        """
        if isinstance(locality_state, str):  # If input is a single string
            if "_" in locality_state:
                locality, state = locality_state.split("_", 1)
                return f"{locality}, {state}" if state.lower() != "unknown" else locality
            else:
                return locality_state  # Return as-is if no '_' in string
        elif isinstance(locality_state, list):  # If input is a list
            formatted_list = []
            for cs in locality_state:
                if "_" in cs:
                    locality, state = cs.split("_", 1)
                    if state.lower() != "unknown":
                        formatted_list.append(f"{locality}, {state}")
                    else:
                        formatted_list.append(locality)
                else:
                    formatted_list.append(cs)  # Append as-is if no '_'
            return formatted_list
        else:
            raise TypeError("Input must be a string or a list of strings.")

    def country_averages_from_nested(self, var_dict, df_mapping: pl.DataFrame):
        """Aggregates nested locality-level values into country-level averages by condition.

        This method processes a dictionary where keys are formatted as
        `"{locality}_{lat}_{long}_{condition}"` and values are dictionaries of measurements.
        It groups all measurements by `country_condition` (e.g., `"USA_0"`, `"USA_1"`)
        and computes the average for each group.

        Args:
            var_dict (dict): Dictionary of locality-condition data, where:
                - Keys (str): "{locality}_{lat}_{long}_{condition}".
                - Values (dict): Inner dictionary of measurements for that locality-condition.
            df_mapping (pd.DataFrame): Mapping DataFrame containing locality-to-country info.

        Returns:
            dict: A dictionary mapping "country_condition" → average value (float).

        Example:
            >>> var_dict = {
            ...     "Paris_48.85_2.35_0": {"a": 10, "b": 20},
            ...     "Paris_48.85_2.35_1": {"a": 30, "b": 50}
            ... }
            >>> df_mapping
               locality     lat  country
            0  Paris  48.85  France
            >>> country_averages_from_nested(var_dict, df_mapping)
            {'France_0': 15.0, 'France_1': 40.0}
        """
        # Store all values grouped by country and condition (e.g., "France_0")
        country_condition_values = defaultdict(list)

        for k, inner_dict in var_dict.items():
            locality, lat, long, condition = k.rsplit("_", 3)
            lat_f = float(lat)
            condition_i = int(condition)

            country = metadata_class.get_value(df_mapping, "locality", locality, "lat", lat_f, "country")
            if country:
                key = f"{country}_{condition_i}"
                country_condition_values[key].extend(inner_dict.values())

        return {key: sum(vals) / len(vals) for key, vals in country_condition_values.items() if vals}

    def country_averages_from_flat(self, var_dict, df_mapping: pl.DataFrame):
        """Aggregates flat locality-level values into country-level averages by condition.

        This method processes a dictionary where keys are formatted as
        `"{locality}_{lat}_{long}_{condition}"` and values are numeric measurements.
        It groups all measurements by `country_condition` (e.g., `"USA_0"`, `"USA_1"`)
        and computes the average for each group.

        Args:
            var_dict (dict): Dictionary of locality-condition values, where:
                - Keys (str): "{locality}_{lat}_{long}_{condition}".
                - Values (float or int): Numeric measurement for that locality-condition.
            df_mapping (pd.DataFrame): Mapping DataFrame containing locality-to-country info.

        Returns:
            dict: A dictionary mapping "country_condition" → average value (float).

        Example:
            >>> var_dict = {
            ...     "Paris_48.85_2.35_0": 10,
            ...     "Paris_48.85_2.35_1": 30
            ... }
            >>> df_mapping
               locality     lat  country
            0  Paris  48.85  France
            >>> country_averages_from_flat(var_dict, df_mapping)
            {'France_0': 10.0, 'France_1': 30.0}
        """
        # Store all values grouped by country and condition (e.g., "France_0")
        country_condition_values = defaultdict(list)

        for k, v in var_dict.items():
            locality, lat, long, condition = k.rsplit("_", 3)
            lat_f = float(lat)
            condition_i = int(condition)

            country = metadata_class.get_value(df_mapping, "locality", locality, "lat", lat_f, "country")
            if country:
                key = f"{country}_{condition_i}"
                country_condition_values[key].append(v)

        return {key: sum(vals) / len(vals) for key, vals in country_condition_values.items() if vals}

    def country_sum_from_cities(self, var_dict, df_mapping: pl.DataFrame):
        """Aggregates locality-level numeric values into country-level sums by condition.

        This method processes a dictionary where keys are formatted as
        `"{locality}_{lat}_{long}_{condition}"` and values are numeric measurements.
        It groups all measurements by `country_condition` (e.g., `"USA_0"`, `"USA_1"`)
        and computes the sum for each group.

        Args:
            var_dict (dict): Dictionary of locality-condition values, where:
                - Keys (str): "{locality}_{lat}_{long}_{condition}".
                - Values (float or int): Numeric measurement for that locality-condition.
            df_mapping (pd.DataFrame): Mapping DataFrame containing locality-to-country info.

        Returns:
            dict: A dictionary mapping "country_condition" → summed value (float or int).

        Example:
            >>> var_dict = {
            ...     "Paris_48.85_2.35_0": 5,
            ...     "Paris_48.85_2.35_1": 7
            ... }
            >>> df_mapping
               locality     lat  country
            0  Paris  48.85  France
            >>> country_sum_from_cities(var_dict, df_mapping)
            {'France_0': 5, 'France_1': 7}
        """
        # Store all values grouped by country and condition (e.g., "France_0")
        country_condition_values = defaultdict(list)

        for k, v in var_dict.items():
            locality, lat, long, condition = k.rsplit("_", 3)
            lat_f = float(lat)
            condition_i = int(condition)

            country = metadata_class.get_value(df_mapping, "locality", locality, "lat", lat_f, "country")
            if country:
                key = f"{country}_{condition_i}"
                country_condition_values[key].append(v)

        return {key: sum(vals) for key, vals in country_condition_values.items()}
