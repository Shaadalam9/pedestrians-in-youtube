import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
import ast
import pandas as pd

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class Values():
    def __init__(self) -> None:
        pass

    def find_values_with_video_id(self, df, key):
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
                - Country
                - GDP per capita
                - Population
                - Population of the country
                - Traffic mortality
                - Continent
                - Literacy rate
                - Average height
                - ISO-3 code for country
        """

        id, start_ = key.rsplit("_", 1)  # Splitting the key into video ID and start time

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Extracting data from the DataFrame row
            video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])
            time_of_day = ast.literal_eval(row["time_of_day"])
            city = row["city"]
            state = row['state'] if not pd.isna(row['state']) else "unknown"
            country = row["country"]
            gdp = row["gmp"]
            population = row["population_city"]
            population_country = row["population_country"]
            traffic_mortality = row["traffic_mortality"]
            continent = row["continent"]
            literacy_rate = row["literacy_rate"]
            avg_height = row["avg_height"]
            iso3 = row["iso3"]
            fps_list = ast.literal_eval(row["fps_list"])

            # Iterate through each video, start time, end time, and time of day
            for video, start, end, time_of_day_, fps in zip(video_ids, start_times, end_times, time_of_day, fps_list):
                # Assume FPS=30 for None
                if not fps:
                    fps = 30
                # Check if the current video matches the specified ID
                if video == id:
                    logger.debug(f"Finding values for {video} start={start}, end={end}")
                    counter = 0
                    # Iterate through each start time
                    for s in start:
                        # Check if the start time matches the specified start time
                        if int(start_) == s:
                            # Calculate gpd per capita to avoid division by zero
                            if population > 0:
                                gpd_capita = gdp/population
                            else:
                                gpd_capita = 0
                            # Return relevant information once found
                            return (video,                      # 0
                                    s,                          # 1
                                    end[counter],               # 2
                                    time_of_day_[counter],      # 3
                                    city,                       # 4
                                    state,                      # 5
                                    country,                    # 6
                                    gpd_capita,                 # 7
                                    population,                 # 8
                                    population_country,         # 9
                                    traffic_mortality,          # 10
                                    continent,                  # 11
                                    literacy_rate,              # 12
                                    avg_height,                 # 13
                                    iso3,                       # 14
                                    fps)                        # 15
                        counter += 1

    def get_value(self, df, column_name1, column_value1, column_name2, column_value2, target_column):
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
        # Treat column_value2 as NaN if it is "unknown"
        if column_value2 == "unknown":
            column_value2 = float('nan')

        # Filter the DataFrame where both conditions are met
        if pd.isna(column_value2):
            result = df[(df[column_name1] == column_value1) & (df[column_name2].isna())][target_column]
        else:
            result = df[(df[column_name1] == column_value1) & (df[column_name2] == column_value2)][target_column]

        # Check if the result is not empty (i.e., if there is a match)
        if not result.empty:
            # Return the first matched value
            return result.values[0]
        else:
            # Return None if no matching value is found
            return None
