import os
import ast
import polars as pl
import common
from utils.core.tools import Tools
from utils.core.metadata import MetaData

metadata_class = MetaData()
tools_class = Tools()


class IO:
    def __init__(self) -> None:
        pass

    def filter_csv_files(self, file: str, df_mapping: pl.DataFrame):
        """
        Filters and processes CSV files based on predefined criteria.

        This function checks if the given file is a CSV, verifies its mapping and value requirements,
        and further processes the file by loading it into a DataFrame and optionally applying geometry corrections.
        Files are only accepted if their mapping indicates sufficient footage and if required columns are present.

        Args:
            file (str): The filename to check and process.

        Returns:
            str or None: The original filename if all checks pass and the file is valid for processing;
                         otherwise, None to indicate the file should be skipped.

        Notes:
            - This method depends on several external classes and variables:
                - `values_class`: For value lookup and calculations.
                - `df_mapping`: DataFrame with mapping data for video IDs.
                - `common`: Configuration utility for various thresholds and flags.
                - `geometry_class`: Utility for geometry correction.
                - `logger`: Logging utility.
                - `folder_path`: Path to search for CSV files.
        """
        # Only process files ending with ".csv"
        file = tools_class.clean_csv_filename(file)
        if file.endswith(".csv"):
            filename = os.path.splitext(file)[0]

            # MetaData helper is assumed Polars-compatible (as per your conversion approach)
            values = metadata_class.find_values_with_video_id(df_mapping, filename)
            if values is None:
                return None

            vehicle_type = values[18]
            vehicle_list = common.get_configs("vehicles_analyse")

            # Only check if list is NOT empty
            if vehicle_list:
                if vehicle_type not in vehicle_list:
                    return None

        return file

    def parse_videos(self, s):
        """Parse a bracketed, comma-separated video string into a list of IDs.

        Args:
            s (str): String representing video IDs, e.g. '[abc,def,ghi]'.

        Returns:
            List[str]: List of video IDs as strings, e.g. ['abc', 'def', 'ghi'].

        Example:
            >>> self.parse_videos('[abc,def]')
            ['abc', 'def']
        """
        if not isinstance(s, str):
            return []
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return [x.strip() for x in s.split(",") if x.strip()]

    def parse_col(self, row, colname):
        """Safely parse a DataFrame row column (stored as a string) to a Python object.

        Args:
            row (pd.Series): The DataFrame row containing the column.
            colname (str): The column name to parse.

        Returns:
            object: The parsed Python object (e.g., list or int). Returns empty list on failure.

        Example:
            >>> self.parse_col(row, 'start_time')
            [[12], [34], [56]]
        """
        try:
            v = row.get(colname)
            if not isinstance(v, str):
                return []
            return ast.literal_eval(v)
        except (ValueError, SyntaxError, KeyError, TypeError, AttributeError):
            return []
