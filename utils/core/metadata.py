import ast
import math
import polars as pl
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger


class MetaData:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _parse_videos_cell(v: str | None) -> list[str]:
        """
        Robustly parse mapping 'videos' cell into list of ids.

        Handles examples:
          [D_LyZL4P3_k]
          "[ikOjqfFj-7o,GDXBW8LRFu4]"
          ["a","b"]
          []
        """
        if not isinstance(v, str):
            return []
        s = v.strip()
        # remove surrounding quotes if present
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        # remove surrounding brackets if present
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]

        parts = []
        for tok in s.split(","):
            t = tok.strip().strip('"').strip("'").strip()
            if t:
                parts.append(t)
        return parts

    @staticmethod
    def _safe_literal_eval(v: str | None):
        if not isinstance(v, str) or not v.strip():
            return None
        try:
            return ast.literal_eval(v)
        except Exception:
            return None

    @staticmethod
    def _state_or_unknown(state_val) -> str:
        if state_val is None:
            return "unknown"
        s = str(state_val).strip()
        if not s or s.lower() == "nan" or s == "NA":
            return "unknown"
        return s

    @staticmethod
    def _eq_expr(colname: str, value) -> pl.Expr:
        """Type-aware equality expression to reduce mismatches (str/int/float)."""
        if value is None:
            return pl.col(colname).is_null()
        if isinstance(value, float):
            if math.isnan(value):
                return pl.col(colname).is_null() | pl.col(colname).is_nan()
            return pl.col(colname).cast(pl.Float64, strict=False) == pl.lit(float(value))
        if isinstance(value, int):
            return pl.col(colname).cast(pl.Int64, strict=False) == pl.lit(int(value))
        # fallback string compare
        return pl.col(colname).cast(pl.Utf8, strict=False) == pl.lit(str(value))

    def find_values_with_video_id(self, df: pl.DataFrame, key: str):
        """Extracts relevant data from a DataFrame based on a given key.

        Args:
            df (DataFrame): The DataFrame containing the data.
            key (str): The key to search for in the DataFrame.

        Returns:
            tuple: A tuple containing information related to the key, including:
                - Video ID
                - Start time
                - End time
                - Time of day
                - City
                - State
                - Latitude
                - Longitude
                - Country
                - GDP per capita
                - Population
                - Population of the country
                - Traffic mortality
                - Continent
                - Literacy rate
                - Average height
                - ISO-3 code for country
                - Fps of the video
                - Type of vehicle
        """
        vid, start_str, fps_str = key.rsplit("_", 2)

        # Iterate rows (Polars)
        for row in df.iter_rows(named=True):
            # Extracting data from the mapping row
            video_ids = self._parse_videos_cell(row.get("videos"))
            start_times = self._safe_literal_eval(row.get("start_time"))
            end_times = self._safe_literal_eval(row.get("end_time"))
            time_of_day = self._safe_literal_eval(row.get("time_of_day"))
            vehicle_type = self._safe_literal_eval(row.get("vehicle_type"))

            if not (
                isinstance(start_times, list)
                and isinstance(end_times, list)
                and isinstance(time_of_day, list)
                and isinstance(vehicle_type, list)
            ):
                continue

            city = row.get("city")
            state = self._state_or_unknown(row.get("state"))
            latitude = row.get("lat")
            longitude = row.get("lon")
            country = row.get("country")
            gdp = row.get("gmp")
            population = row.get("population_city")
            population_country = row.get("population_country")
            traffic_mortality = row.get("traffic_mortality")
            continent = row.get("continent")
            literacy_rate = row.get("literacy_rate")
            avg_height = row.get("avg_height")
            iso3 = row.get("iso3")

            # Iterate through each video, start time list, end time list, etc.
            for video, start_list, end_list, tod_list, vtype_list in zip(
                video_ids, start_times, end_times, time_of_day, vehicle_type
            ):
                if video != vid:
                    continue

                if not isinstance(start_list, list) or not isinstance(end_list, list) or not isinstance(tod_list, list):  # noqa: E501
                    continue

                logger.debug(f"Finding values for {video} start={start_list}, end={end_list}")

                try:
                    start_target = int(start_str)
                except Exception:
                    continue

                for idx, s in enumerate(start_list):
                    try:
                        if int(s) != start_target:
                            continue
                    except Exception:
                        continue

                    # GDP per capita (avoid div-by-zero)
                    try:
                        pop_i = int(population) if population is not None else 0
                    except Exception:
                        pop_i = 0
                    try:
                        gdp_i = int(gdp) if gdp is not None else 0
                    except Exception:
                        gdp_i = 0

                    gpd_capita = (gdp_i / pop_i) if pop_i > 0 else 0

                    end_val = end_list[idx] if idx < len(end_list) else None
                    tod_val = tod_list[idx] if idx < len(tod_list) else None

                    # Return relevant information once found (same positional tuple)
                    return (
                        video,                # 0
                        s,                    # 1
                        end_val,              # 2
                        tod_val,              # 3
                        city,                 # 4
                        state,                # 5
                        latitude,             # 6
                        longitude,            # 7
                        country,              # 8
                        gpd_capita,           # 9
                        population,           # 10
                        population_country,   # 11
                        traffic_mortality,    # 12
                        continent,            # 13
                        literacy_rate,        # 14
                        avg_height,           # 15
                        iso3,                 # 16
                        int(fps_str),         # 17
                        vtype_list,           # 18
                    )

        return None

    def get_value(self, df: pl.DataFrame, column_name1: str, column_value1,
                  column_name2: str | None, column_value2, target_column: str):
        """
        Retrieves a value from the target_column based on the condition
        that both column_name1 matches column_value1 and column_name2 matches column_value2.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the mapping file.
        column_name1 (str): The first column to search for the matching value.
        column_value1 (str): The value to search for in column_name1.
        column_name2 (str): The second column to search for the matching value.
        column_value2 (str): The value to search for in column_name2. If "unknown", the value is treated as NaN.
        target_column (str): The column from which to retrieve the corresponding value.

        Returns:
        Any: The value from target_column that corresponds to the matching values in both
             column_name1 and column_name2.
        """
        if column_name2 is None or column_value2 is None:
            out = (
                df.filter(self._eq_expr(column_name1, column_value1)).select(target_column).head(1)
            )
            return out.item(0, target_column) if out.height > 0 else None

        # Treat "unknown" as NULL
        if isinstance(column_value2, str) and column_value2 == "unknown":
            column_value2 = None

        # Treat NaN as NULL-like
        if isinstance(column_value2, float) and math.isnan(column_value2):
            column_value2 = None

        filt = self._eq_expr(column_name1, column_value1) & self._eq_expr(column_name2, column_value2)

        out = df.filter(filt).select(target_column).head(1)
        return out.item(0, target_column) if out.height > 0 else None
