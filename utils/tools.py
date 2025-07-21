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
