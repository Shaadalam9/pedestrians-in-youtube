import ast


class Tools():
    def __init__(self) -> None:
        pass

    def count_unique_channels(self, channel_str):
        """Counts the number of unique channel names from a string representation.

        This method handles both:
            1. A string representation of a Python list (e.g., `"['A', 'B', 'A']"`).
            2. A comma-separated string (e.g., `"A, B, A"`).

        The method strips extra whitespace, removes duplicates, and returns
        the count of unique channels.

        Args:
            channel_str (str): String containing channel names, either in list-like
                format or as a comma-separated string.

        Returns:
            int: The number of unique channel names.

        Examples:
            >>> count_unique_channels("['A', 'B', 'A']")
            2
            >>> count_unique_channels("A, B, A")
            2
            >>> count_unique_channels("[]")
            0
        """
        try:
            # Attempt to safely evaluate the string into a Python object
            channel_list = ast.literal_eval(channel_str)

            # If the evaluated result is not a list, fall back to comma-split
            if not isinstance(channel_list, list):
                channel_list = channel_str.strip("[]").split(",")

            # Strip whitespace from each channel name
            channel_list = [ch.strip() for ch in channel_list]

            # Return count of unique channels
            return len(set(channel_list))

        except Exception:
            # Fallback: treat input as a plain comma-separated string
            channel_list = channel_str.strip("[]").split(",")
            channel_list = [ch.strip() for ch in channel_list]
            return len(set(channel_list))

    def compute_avg_variable_city(self, variable_city):
        """Computes the average value for each city-condition entry in a nested dictionary.

        This method processes a dictionary where each key represents a city-condition
        (e.g., "{city}_{lat}_{long}_{condition}") and each value is another dictionary
        containing numeric measurements. It calculates the average of the inner
        dictionary's values and stores it in a new dictionary under the same key.

        Args:
            variable_city (dict): Dictionary of city-condition data, where:
                - Keys (str): "{city}_{lat}_{long}_{condition}".
                - Values (dict): Inner dictionary containing numeric measurements.

        Returns:
            dict: A dictionary with the same keys as `variable_city`, but with
            average values (float) instead of dictionaries.

        Example:
            >>> variable_city = {
            ...     "Paris_48.85_2.35_0": {"a": 10, "b": 20},
            ...     "Paris_48.85_2.35_1": {"a": 30, "b": 50}
            ... }
            >>> compute_avg_variable_city(variable_city)
            {
                "Paris_48.85_2.35_0": 15.0,
                "Paris_48.85_2.35_1": 40.0
            }
        """
        avg_dict = {}

        # Loop through each city-condition key and its inner dictionary
        for key, inner_dict in variable_city.items():
            # Extract the numeric values from the inner dictionary
            values = list(inner_dict.values())

            # Compute the average, handling the case of empty value lists
            avg_value = sum(values) / len(values) if values else 0

            # Store the computed average in the result dictionary
            avg_dict[key] = avg_value

        return avg_dict

    def clean_csv_filename(self, file):
        """Cleans and normalizes a CSV filename string.

        If the provided filename ends with `.csv`, it is returned unchanged.
        Otherwise:
            1. Leading dots are removed (e.g., ".filename" → "filename").
            2. If ".csv" appears anywhere in the string, the name is truncated
               immediately after ".csv".
            3. If ".csv" is not present, the cleaned base name is returned as-is.

        Args:
            file (str): The input filename or path to be cleaned.

        Returns:
            str: A cleaned and normalized CSV filename.

        Examples:
            >>> clean_csv_filename("data.csv")
            'data.csv'
            >>> clean_csv_filename(".hiddenfile.csv.backup")
            'hiddenfile.csv'
            >>> clean_csv_filename("report_file")
            'report_file'
        """
        # If already a proper CSV filename, return unchanged
        if file.endswith(".csv"):
            return file

        # Remove leading dot if present (e.g., hidden files in Unix)
        file_clean = file.lstrip(".")

        # Find the position of ".csv" in the cleaned name
        csv_pos = file_clean.find(".csv")

        if csv_pos != -1:
            # Keep everything up to and including ".csv"
            base_name = file_clean[:csv_pos + 4]
        else:
            # If ".csv" not found, just return the cleaned name
            base_name = file_clean

        return base_name
