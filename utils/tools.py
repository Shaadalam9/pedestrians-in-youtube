import ast


class Tools():
    def __init__(self) -> None:
        pass

    def count_unique_channels(self, channel_str):
        # Convert string to list (if it's not already a list)
        # Remove any leading/trailing spaces and square brackets
        try:
            # Convert the string to an actual list
            channel_list = ast.literal_eval(channel_str)

            # If the conversion didn't work, channel_list may not be a list
            if not isinstance(channel_list, list):
                channel_list = channel_str.strip('[]').split(',')

            # Strip whitespace and count unique elements
            channel_list = [ch.strip() for ch in channel_list]
            return len(set(channel_list))
        except Exception:
            # Fallback: count after splitting by comma
            channel_list = channel_str.strip('[]').split(',')
            channel_list = [ch.strip() for ch in channel_list]
            return len(set(channel_list))

    def compute_avg_variable_city(self, variable_city):
        """
        Compute the average value for each city-condition key in a nested dictionary.
        """
        avg_dict = {}

        for key, inner_dict in variable_city.items():
            # Compute average
            values = list(inner_dict.values())
            avg_value = sum(values) / len(values) if values else 0

            # Assign average directly to the same key
            avg_dict[key] = avg_value

        return avg_dict

    def clean_csv_filename(self, file):
        """
        If the filename ends with '.csv', returns it as-is.
        Otherwise:
        - Removes leading dot
        - Truncates at first '.csv' if present
        - Else returns cleaned filename
        """
        if file.endswith('.csv'):
            return file
        file_clean = file.lstrip('.')  # Remove leading dot if present
        csv_pos = file_clean.find('.csv')
        if csv_pos != -1:
            base_name = file_clean[:csv_pos + 4]  # includes ".csv"
        else:
            base_name = file_clean  # fallback if '.csv' not found
        return base_name
