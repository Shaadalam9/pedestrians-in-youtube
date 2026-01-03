import common
import pandas as pd
import numpy as np
import plotly.express as px
from custom_logger import CustomLogger
from utils.plotting.io import IO
from utils.core.grouping import Grouping
from utils.core.metadata import MetaData
from utils.core.tools import Tools

metadata_class = MetaData()
plots_io_class = IO()
grouping_class = Grouping()
tools_class = Tools()
logger = CustomLogger(__name__)  # use custom logger


class Correlations:
    def __init__(self) -> None:
        pass

    def correlation_matrix(self, df_mapping, ped_cross_city, person_city, bicycle_city, car_city,
                           motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city, cellphone_city,
                           trf_sign_city, speed_values, time_values, avg_time, avg_speed):
        """
        Compute and visualise correlation matrices for various city-level traffic and demographic data.

        This method:
        - Loads precomputed statistical data from a pickled file.
        - Aggregates metrics like speed, time, vehicle/pedestrian counts, and socioeconomic indicators.
        - Constructs structured dictionaries for day (condition 0) and night (condition 1) conditions.
        - Computes Spearman correlation matrices for:
            - Daytime data
            - Nighttime data
            - Averaged across both conditions
            - Per continent basis
        - Generates and saves Plotly heatmaps for all computed correlation matrices.

        Args:
            df_mapping (pd.DataFrame): A mapping DataFrame containing metadata for each city-state combination,
                                       including country, continent, GDP, literacy rate, etc.

        Raises:
            ValueError: If essential data (e.g., average speed or time) is missing.
        """
        logger.info("Plotting correlation matrices.")
        final_dict = {}

        # Check if both 'speed' and 'time' are valid dictionaries
        if avg_speed is None or avg_time is None:
            raise ValueError("Either 'speed' or 'time' returned None, please check the input data or calculations.")

        # Remove the ones where there is data missing for a specific country and condition
        common_keys = avg_speed.keys() & avg_time.keys()

        # Retain only the key-value pairs where the key is present in both dictionaries
        avg_speed = {key: avg_speed[key] for key in common_keys}
        avg_time = {key: avg_time[key] for key in common_keys}

        # Now populate the final_dict with city-wise data
        for city_condition, speed in avg_speed.items():
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            continent = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "continent")
            population_country = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "population_country")  # noqa: E501
            gdp_city = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "gmp")
            traffic_mortality = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "traffic_mortality")  # noqa: E501
            literacy_rate = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "literacy_rate")
            gini = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "gini")
            traffic_index = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "traffic_index")

            if country or iso_code is not None:

                # Initialise the city's dictionary if not already present
                if f'{city}_{lat}_{long}' not in final_dict:
                    final_dict[f'{city}_{lat}_{long}'] = {
                        "avg_speed_0": None, "avg_speed_1": None, "avg_time_0": None, "avg_time_1": None,
                        "speed_val_0": None, "speed_val_1": None, "time_val_0": None, "time_val_1": None,
                        "ped_cross_city_0": 0, "ped_cross_city_1": 0,
                        "person_city_0": 0, "person_city_1": 0, "bicycle_city_0": 0,
                        "bicycle_city_1": 0, "car_city_0": 0, "car_city_1": 0,
                        "motorcycle_city_0": 0, "motorcycle_city_1": 0, "bus_city_0": 0,
                        "bus_city_1": 0, "truck_city_0": 0, "truck_city_1": 0,
                        "cross_evnt_city_0": 0, "cross_evnt_city_1": 0, "vehicle_city_0": 0,
                        "vehicle_city_1": 0, "cellphone_city_0": 0, "cellphone_city_1": 0,
                        "trf_sign_city_0": 0, "trf_sign_city_1": 0,
                    }

                # Populate the corresponding speed and time based on the condition
                final_dict[f'{city}_{lat}_{long}'][f"avg_speed_{condition}"] = speed
                if f'{city}_{lat}_{long}_{condition}' in avg_time:
                    final_dict[f'{city}_{lat}_{long}'][f"avg_time_{condition}"] = avg_time.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"time_val_{condition}"] = time_values.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"speed_val_{condition}"] = speed_values.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"time_val_{condition}"] = time_values.get(
                        f'{city}_{lat}_{long}_{condition}', None)
                    final_dict[f'{city}_{lat}_{long}'][f"ped_cross_city_{condition}"] = ped_cross_city.get(
                        f'{city}_{lat}_{long}_{condition}', None)

                    avg_person_city = tools_class.compute_avg_variable_city(person_city)
                    final_dict[f'{city}_{lat}_{long}'][f"person_city_{condition}"] = avg_person_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_bicycle_city = tools_class.compute_avg_variable_city(bicycle_city)
                    final_dict[f'{city}_{lat}_{long}'][f"bicycle_city_{condition}"] = avg_bicycle_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_car_city = tools_class.compute_avg_variable_city(car_city)
                    final_dict[f'{city}_{lat}_{long}'][f"car_city_{condition}"] = avg_car_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_motorcycle_city = tools_class.compute_avg_variable_city(motorcycle_city)
                    final_dict[f'{city}_{lat}_{long}'][f"motorcycle_city_{condition}"] = avg_motorcycle_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_bus_city = tools_class.compute_avg_variable_city(bus_city)
                    final_dict[f'{city}_{lat}_{long}'][f"bus_city_{condition}"] = avg_bus_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_truck_city = tools_class.compute_avg_variable_city(truck_city)
                    final_dict[f'{city}_{lat}_{long}'][f"truck_city_{condition}"] = avg_truck_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    final_dict[f'{city}_{lat}_{long}'][f"cross_evnt_city_{condition}"] = cross_evnt_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_vehicle_city = tools_class.compute_avg_variable_city(vehicle_city)
                    final_dict[f'{city}_{lat}_{long}'][f"vehicle_city_{condition}"] = avg_vehicle_city.get(
                        f'{city}_{lat}_{long}_{condition}', None)

                    avg_cellphone_city = tools_class.compute_avg_variable_city(cellphone_city)
                    final_dict[f'{city}_{lat}_{long}'][f"cellphone_city_{condition}"] = avg_cellphone_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    avg_trf_sign_city = tools_class.compute_avg_variable_city(trf_sign_city)
                    final_dict[f'{city}_{lat}_{long}'][f"trf_sign_city_{condition}"] = avg_trf_sign_city.get(
                        f'{city}_{lat}_{long}_{condition}', 0)

                    final_dict[f'{city}_{lat}_{long}'][f"traffic_mortality_{condition}"] = traffic_mortality
                    final_dict[f'{city}_{lat}_{long}'][f"literacy_rate_{condition}"] = literacy_rate
                    final_dict[f'{city}_{lat}_{long}'][f"gini_{condition}"] = gini
                    final_dict[f'{city}_{lat}_{long}'][f"traffic_index_{condition}"] = traffic_index
                    final_dict[f'{city}_{lat}_{long}'][f"continent_{condition}"] = continent
                    if gdp_city is not None:
                        final_dict[f'{city}_{lat}_{long}'][f"gmp_{condition}"] = gdp_city/population_country

        # Initialise an empty list to store the rows for the DataFrame
        data_day, data_night = [], []

        # Loop over each city and gather relevant values for condition 0
        for city in final_dict:
            # Initialise a dictionary for the row
            row_day, row_night = {}, {}

            # Add data for condition 0 (ignore 'speed_val' and 'time_val')
            for condition in ['0']:  # Only include condition 0
                for key, value in final_dict[city].items():
                    if condition in key and 'speed_val' not in key and 'time_val' not in key and 'continent' not in key:  # noqa:E501
                        row_day[key] = value

            # Append the row to the data list
            data_day.append(row_day)

            for condition in ['1']:  # Only include condition 1
                for key, value in final_dict[city].items():
                    if condition in key and 'speed_val' not in key and 'time_val' not in key and 'continent' not in key:  # noqa:E501
                        row_night[key] = value

            # Append the row to the data list
            data_night.append(row_night)

        # Convert the list of rows into a Pandas DataFrame
        df_day = pd.DataFrame(data_day)
        df_night = pd.DataFrame(data_night)

        # Calculate the correlation matrix
        corr_matrix_day = df_day.corr(method='spearman')
        corr_matrix_night = df_night.corr(method='spearman')

        # Rename the variables in the correlation matrix
        rename_dict_1 = {
            'avg_speed_0': 'Speed of', 'avg_speed_1': 'Crossing speed',
            'avg_time_0': 'Crossing initiation time', 'avg_time_1': 'Crossing initiation time',
            'ped_cross_city_0': 'Crossing', 'ped_cross_city_1': 'Crossing',
            'person_city_0': 'Detected persons', 'person_city_1': 'Detected persons',
            'bicycle_city_0': 'Detected bicycles', 'bicycle_city_1': 'Detected bicycles',
            'car_city_0': 'Detected cars', 'car_city_1': 'Detected cars',
            'motorcycle_city_0': 'Detected motorcycles', 'motorcycle_city_1': 'Detected motorcycles',
            'bus_city_0': 'Detected buses', 'bus_city_1': 'Detected buses',
            'truck_city_0': 'Detected trucks', 'truck_city_1': 'Detected trucks',
            'cross_evnt_city_0': 'Crossings without traffic lights',
            'cross_evnt_city_1': 'Crossings without traffic lights',
            'vehicle_city_0': 'Detected motor vehicles',
            'vehicle_city_1': 'Detected motor vehicles',
            'cellphone_city_0': 'Detected cellphones', 'cellphone_city_1': 'Detected cellphones',
            'trf_sign_city_0': 'Detected traffic signs', 'trf_sign_city_1': 'Detected traffic signs',
            'gmp_0': 'GMP', 'gmp_1': 'GMP',
            'traffic_mortality_0': 'Traffic mortality', 'traffic_mortality_1': 'Traffic mortality',
            'literacy_rate_0': 'Literacy rate', 'literacy_rate_1': 'Literacy rate',
            'gini_0': 'Gini coefficient', 'gini_1': 'Gini coefficient', 'traffic_index_0': 'Traffic index',
            'traffic_index_1': 'Traffic index'
            }

        corr_matrix_day = corr_matrix_day.rename(columns=rename_dict_1, index=rename_dict_1)
        corr_matrix_night = corr_matrix_night.rename(columns=rename_dict_1, index=rename_dict_1)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_day, text_auto=".2f",  # Display correlation values on the heatmap # type: ignore
                        color_continuous_scale='RdBu',  # Color scale
                        aspect="auto")  # Automatically adjust aspect ratio
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        plots_io_class.save_plotly_figure(fig, "correlation_matrix_heatmap_day", save_final=True)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_night, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation Matrix Heatmap in night"  # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        plots_io_class.save_plotly_figure(fig, "correlation_matrix_heatmap_night", save_final=True)

        # Initialise a list to store rows of data (one row per city)
        data_rows = []

        # Assuming `conditions` is a list of conditions working with
        conditions = ['0', '1']  # Modify this list to include all condition (e.g., '0', '1', etc.)

        # Iterate over each city and condition
        for city in final_dict:
            # Initialise a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'gmp', 'traffic_mortality', 'literacy_rate', 'gini', 'traffic_index']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[city].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[city].get(f"speed_val_{condition}", [])
                time_vals = final_dict[city].get(f"time_val_{condition}", [])

                if speed_vals:  # Avoid division by zero or empty dict
                    all_speed_values = [val for inner_dict in speed_vals.values() for val in inner_dict.values()]
                    if all_speed_values:  # Check to avoid computing mean on empty list
                        row_data[f"avg_speed_val_{condition}"] = np.mean(all_speed_values)
                    else:
                        row_data[f"avg_speed_val_{condition}"] = np.nan
                else:
                    row_data[f"avg_speed_val_{condition}"] = np.nan  # Handle empty or missing dict

                if time_vals:
                    all_time_values = [val for inner_dict in time_vals.values() for val in inner_dict.values()]
                    if all_time_values:
                        row_data[f"avg_time_val_{condition}"] = np.mean(all_time_values)
                    else:
                        row_data[f"avg_time_val_{condition}"] = np.nan
                else:
                    row_data[f"avg_time_val_{condition}"] = np.nan  # Handle empty or missing dict

            # Append the row data for the current city
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)

        # Create a new DataFrame to average the columns across conditions
        agg_df = pd.DataFrame()

        # Loop through the columns in the original DataFrame
        for col in df.columns:
            # Extract the feature name (without condition part)
            feature_name = "_".join(col.split("_")[:-1])
            condition = col.split("_")[-1]

            # Create a new column by averaging values across conditions for the same feature
            if feature_name not in agg_df.columns:
                # Select the columns for this feature across all conditions
                condition_cols = [c for c in df.columns if feature_name in c]
                agg_df[feature_name] = df[condition_cols].mean(axis=1)

        # Compute the correlation matrix on the aggregated DataFrame
        corr_matrix_avg = agg_df.corr(method='spearman')

        # Rename the variables in the correlation matrix (example: renaming keys)
        rename_dict_2 = {
            'avg_speed_val': 'Crossing speed', 'avg_time_val': 'Crossing initiation time',
            'ped_cross_city': 'Crossing', 'person_city': 'Detected persons',
            'bicycle_city': 'Detected bicycles', 'car_city': 'Detected cars',
            'motorcycle_city': 'Detected motorcycles', 'bus_city': 'Detected buses',
            'truck_city': 'Detected trucks', 'cross_evnt_city': 'Crossings without traffic light',
            'vehicle_city': 'Detected all motor vehicles', 'cellphone_city': 'Detected cellphones',
            'trf_sign_city': 'Detected traffic signs', 'gmp_city': 'GMP',
            'traffic_mortality_city': 'Traffic mortality', 'literacy_rate_city': 'Literacy rate',
            'gini': 'Gini coefficient', 'traffic_index': 'Traffic Index'
            }

        corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_2, index=rename_dict_2)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation matrix heatmap averaged" # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # use value from config file
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        plots_io_class.save_plotly_figure(fig, "correlation_matrix_heatmap_averaged", save_final=True)

        # Continent Wise

        # Initialise a list to store rows of data (one row per city)
        data_rows = []

        # Assuming `conditions` is a list of conditions working with
        conditions = ['0', '1']  # Modify this list to include all conditions (e.g., '0', '1', etc.)
        unique_continents = df_mapping['continent'].unique()

        # Iterate over each city and condition
        for city in final_dict:
            # Initialise a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'gmp', 'traffic_mortality', 'literacy_rate', 'continent', 'gini', 'traffic_index']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[city].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[city].get(f"speed_val_{condition}", [])
                time_vals = final_dict[city].get(f"time_val_{condition}", [])

                if speed_vals:
                    all_speed_values = [val for inner_dict in speed_vals.values() for val in inner_dict.values()]
                    row_data[f"avg_speed_val_{condition}"] = np.mean(all_speed_values) if all_speed_values else np.nan
                else:
                    row_data[f"avg_speed_val_{condition}"] = np.nan

                # Handle avg_time_val
                if time_vals:
                    all_time_values = [val for inner_dict in time_vals.values() for val in inner_dict.values()]
                    row_data[f"avg_time_val_{condition}"] = np.mean(all_time_values) if all_time_values else np.nan
                else:
                    row_data[f"avg_time_val_{condition}"] = np.nan

            # Append the row data for the current city
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)

        for continents in unique_continents:
            filtered_df = df[(df['continent_0'] == continents) | (df['continent_1'] == continents)]
            # Create a new DataFrame to average the columns across conditions
            agg_df = pd.DataFrame()

            # Loop through the columns in the original DataFrame
            for col in filtered_df.columns:
                # Extract the feature name (without condition part)
                feature_name = "_".join(col.split("_")[:-1])
                condition = col.split("_")[-1]

                # Skip columns named "continent_0" or "continent_1"
                if "continent" in feature_name:
                    continue

                # Create a new column by averaging values across conditions for the same feature
                if feature_name not in agg_df.columns:
                    # Select the columns for this feature across all conditions
                    condition_cols = [c for c in filtered_df.columns if feature_name in c]
                    agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)

            # Compute the correlation matrix on the aggregated DataFrame
            corr_matrix_avg = agg_df.corr(method='spearman')

            # Rename the variables in the correlation matrix (example: renaming keys)
            rename_dict_3 = {
                'avg_speed_val': 'Crossing speed', 'avg_time_val': 'Crossing initiation time',
                'ped_cross_city': 'Crossing', 'person_city': 'Detected persons',
                'bicycle_city': 'Detected bicycles', 'car_city': 'Detected cars',
                'motorcycle_city': 'Detected motorcycles', 'bus_city': 'Detected buses',
                'truck_city': 'Detected trucks', 'cross_evnt_city': 'Crossings without traffic light',
                'vehicle_city': 'Detected all motor vehicles', 'cellphone_city': 'Detected cellphones',
                'trf_sign_city': 'Detected traffic signs', 'gmp': 'GMP',
                'traffic_mortality': 'Traffic mortality', 'literacy_rate': 'Literacy rate', 'gini': 'Gini coefficient',
                'traffic_index': 'Traffic Index'
                }

            corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_3, index=rename_dict_3)

            # Generate the heatmap using Plotly
            fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                            color_continuous_scale='RdBu',  # Color scale
                            aspect="auto",  # Automatically adjust aspect ratio
                            # title=f"Correlation matrix heatmap {continents}"  # Title of the heatmap
                            )

            fig.update_layout(coloraxis_showscale=False)

            # update font family
            fig.update_layout(font=dict(family=common.get_configs('font_family')))

            # use value from config file
            fig.update_layout(font=dict(size=common.get_configs('font_size')))

            plots_io_class.save_plotly_figure(fig, f"correlation_matrix_heatmap_{continents}", save_final=True)

    def correlation_matrix_country(self, df_mapping, df_countries, ped_cross_city, person_city, bicycle_city, car_city,
                                   motorcycle_city, bus_city, truck_city, cross_evnt_city, vehicle_city,
                                   cellphone_city, trf_sign_city, avg_speed_country, avg_time_country,
                                   cross_no_equip_country, save_file=True):

        """Generates and saves correlation matrices of traffic and demographic data by country.

        This method:
            1. Aggregates raw detection data from city level to country level.
            2. Combines average speeds, times, and various detection counts.
            3. Produces correlation matrices for:
                - Daytime only
                - Nighttime only
                - Averaged across both conditions
                - Per continent

        Correlations are computed using Spearman's rank method and visualized as Plotly heatmaps.

        Args:
            df_mapping (pd.DataFrame): Mapping of cities to countries, ISO codes, continents, etc.
            df_countries (pd.DataFrame): Country-level dataset containing aggregate measures.
            ped_cross_city (dict): Counts of pedestrian crossings by city.
            person_city (dict): Counts of detected persons by city.
            bicycle_city (dict): Counts of detected bicycles by city.
            car_city (dict): Counts of detected cars by city.
            motorcycle_city (dict): Counts of detected motorcycles by city.
            bus_city (dict): Counts of detected buses by city.
            truck_city (dict): Counts of detected trucks by city.
            cross_evnt_city (dict): Counts of crossing events without traffic lights by city.
            vehicle_city (dict): Counts of all motor vehicles by city.
            cellphone_city (dict): Counts of detected cellphones by city.
            trf_sign_city (dict): Counts of detected traffic signs by city.
            avg_speed_country (dict): Average crossing speeds per country-condition key.
            avg_time_country (dict): Average crossing initiation times per country-condition key.
            cross_no_equip_country (dict): Counts of crossings without equipment by country.
            save_file (bool, optional): If True, saves output figures; if False, shows them interactively.

        Raises:
            ValueError: If `avg_speed_country` or `avg_time_country` is None.

        Returns:
            None: Figures are displayed or saved; no explicit return value.
        """

        logger.info("Plotting correlation matrices.")
        final_dict = {}

        # Aggregate city-level counts to country level
        ped_cross_city = grouping_class.country_sum_from_cities(ped_cross_city, df_mapping)
        person_city = grouping_class.country_averages_from_nested(person_city, df_mapping)
        bicycle_city = grouping_class.country_averages_from_nested(bicycle_city, df_mapping)
        car_city = grouping_class.country_averages_from_nested(car_city, df_mapping)
        motorcycle_city = grouping_class.country_averages_from_nested(motorcycle_city, df_mapping)
        bus_city = grouping_class.country_averages_from_nested(bus_city, df_mapping)
        truck_city = grouping_class.country_averages_from_nested(truck_city, df_mapping)
        vehicle_city = grouping_class.country_averages_from_nested(vehicle_city, df_mapping)
        cellphone_city = grouping_class.country_averages_from_nested(cellphone_city, df_mapping)
        trf_sign_city = grouping_class.country_averages_from_nested(trf_sign_city, df_mapping)
        cross_evnt_city = grouping_class.country_averages_from_flat(cross_evnt_city, df_mapping)

        # Validate required inputs
        if avg_speed_country is None or avg_time_country is None:
            raise ValueError("Either 'speed' or 'time' returned None, please check the input data or calculations.")

        # Keep only countries with both speed and time data
        common_keys = avg_speed_country.keys() & avg_time_country.keys()

        # Retain only the key-value pairs where the key is present in both dictionaries
        avg_speed_country = {key: avg_speed_country[key] for key in common_keys}
        avg_time_country = {key: avg_time_country[key] for key in common_keys}

        # Step 4: Populate `final_dict` with aggregated metrics per country-condition
        for country_condition, speed in avg_speed_country.items():
            country, condition = country_condition.split('_')

            # Key format is "<country>_<condition>". Extract both parts, then fetch
            # country-level metadata from df_mapping/df_countries for enrichment
            iso_code = metadata_class.get_value(df_mapping, "country", country, None, None, "iso3")
            continent = metadata_class.get_value(df_mapping, "country", country, None, None, "continent")
            traffic_mortality = metadata_class.get_value(df_mapping, "country", country, None,
                                                         None, "traffic_mortality")
            literacy_rate = metadata_class.get_value(df_mapping, "country", country, None, None, "literacy_rate")
            gini = metadata_class.get_value(df_mapping, "country", country, None, None, "gini")
            med_age = metadata_class.get_value(df_mapping, "country", country, None, None, "med_age")
            avg_day_night_speed = metadata_class.get_value(df_countries, "country", country,
                                                           None, None, "speed_crossing_day_night_country_avg")
            avg_day_night_time = metadata_class.get_value(df_countries, "country", country,
                                                          None, None, "time_crossing_day_night_country_avg")

            if country or iso_code is not None:

                # Initialise the city's dictionary if not already present
                if f'{country}' not in final_dict:
                    final_dict[f'{country}'] = {
                                                "avg_speed_0": None,
                                                "avg_speed_1": None,
                                                "avg_time_0": None,
                                                "avg_time_1": None,
                                                "avg_day_night_speed": None,
                                                "avg_day_night_time": None,
                                                "ped_cross_city_0": 0,
                                                "ped_cross_city_1": 0,
                                                "person_city_0": 0,
                                                "person_city_1": 0,
                                                "bicycle_city_0": 0,
                                                "bicycle_city_1": 0,
                                                "car_city_0": 0,
                                                "car_city_1": 0,
                                                "motorcycle_city_0": 0,
                                                "motorcycle_city_1": 0,
                                                "bus_city_0": 0,
                                                "bus_city_1": 0,
                                                "truck_city_0": 0,
                                                "truck_city_1": 0,
                                                "vehicle_city_0": 0,
                                                "vehicle_city_1": 0,
                                                "cellphone_city_0": 0,
                                                "cellphone_city_1": 0,
                                                "trf_sign_city_0": 0,
                                                "trf_sign_city_1": 0,
                                                "cross_evnt_city_0": 0,
                                                "cross_evnt_city_1": 0,
                                                }

                # Populate the corresponding speed and time based on the condition
                final_dict[f'{country}'][f"avg_speed_{condition}"] = speed

                if f'{country}_{condition}' in avg_time_country:
                    final_dict[f'{country}'][f"avg_time_{condition}"] = avg_time_country.get(
                        f'{country}_{condition}', None)
                    final_dict[f'{country}'][f"ped_cross_city_{condition}"] = ped_cross_city.get(
                        f'{country}_{condition}', None)
                    final_dict[f'{country}'][f"person_city_{condition}"] = person_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"bicycle_city_{condition}"] = bicycle_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"car_city_{condition}"] = car_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"motorcycle_city_{condition}"] = motorcycle_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"bus_city_{condition}"] = bus_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"truck_city_{condition}"] = truck_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"vehicle_city_{condition}"] = vehicle_city.get(
                        f'{country}_{condition}', None)
                    final_dict[f'{country}'][f"cellphone_city_{condition}"] = cellphone_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"trf_sign_city_{condition}"] = trf_sign_city.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"cross_evnt_city_{condition}"] = cross_no_equip_country.get(
                        f'{country}_{condition}', 0)

                    final_dict[f'{country}'][f"traffic_mortality_{condition}"] = None if traffic_mortality == 0 else traffic_mortality  # noqa:E501
                    final_dict[f'{country}'][f"literacy_rate_{condition}"] = None if literacy_rate == 0 else literacy_rate  # noqa:E501
                    final_dict[f'{country}'][f"gini_{condition}"] = None if gini == 0 else gini
                    final_dict[f'{country}'][f"med_age_{condition}"] = None if med_age == 0 else med_age
                    final_dict[f'{country}'][f"continent_{condition}"] = continent

                    final_dict[f'{country}']["avg_day_night_speed"] = avg_day_night_speed
                    final_dict[f'{country}']["avg_day_night_time"] = avg_day_night_time

        # Initialise an empty list to store the rows for the DataFrame
        data_day, data_night = [], []

        # Loop over each city and gather relevant values for condition 0
        for country in final_dict:
            # Initialise a dictionary for the row
            row_day, row_night = {}, {}

            # Add data for condition 0 (ignore 'speed_val' and 'time_val')
            for condition in ['0']:  # Only include condition 0
                for key, value in final_dict[country].items():
                    if (
                        condition in key
                        and 'speed_val' not in key
                        and 'time_val' not in key
                        and 'continent' not in key
                        and 'avg_day_night_speed' not in key
                        and 'avg_day_night_time' not in key
                    ):
                        row_day[key] = value

            # Append the row to the data list
            data_day.append(row_day)

            for condition in ['1']:  # Only include condition 1
                for key, value in final_dict[country].items():
                    if (
                        condition in key
                        and 'speed_val' not in key
                        and 'time_val' not in key
                        and 'continent' not in key
                        and 'avg_day_night_speed' not in key
                        and 'avg_day_night_time' not in key
                    ):
                        row_night[key] = value

            # Append the row to the data list
            data_night.append(row_night)

        # Convert the list of rows into a Pandas DataFrame
        df_day = pd.DataFrame(data_day)
        df_night = pd.DataFrame(data_night)

        # Calculate the correlation matrix
        corr_matrix_day = df_day.corr(method='spearman')
        corr_matrix_night = df_night.corr(method='spearman')

        # Rename the variables in the correlation matrix
        rename_dict_1 = {
            'avg_speed_0': 'Crossing speed',
            'avg_speed_1': 'Crossing speed',
            'avg_time_0': 'Crossing initiation time',
            'avg_time_1': 'Crossing initiation time',
            'ped_cross_city_0': 'Detected crossings',
            'ped_cross_city_1': 'Detected crossings',
            'person_city_0': 'Detected persons',
            'person_city_1': 'Detected persons',
            'bicycle_city_0': 'Detected bicycles',
            'bicycle_city_1': 'Detected bicycles',
            'car_city_0': 'Detected cars',
            'car_city_1': 'Detected cars',
            'motorcycle_city_0': 'Detected motorcycles',
            'motorcycle_city_1': 'Detected motorcycles',
            'bus_city_0': 'Detected buses',
            'bus_city_1': 'Detected buses',
            'truck_city_0': 'Detected trucks',
            'truck_city_1': 'Detected trucks',
            'vehicle_city_0': 'Detected all motor vehicles',
            'vehicle_city_1': 'Detected all motor vehicles',
            'cellphone_city_0': 'Detected cellphones',
            'cellphone_city_1': 'Detected cellphones',
            'trf_sign_city_0': 'Detected traffic signs',
            'trf_sign_city_1': 'Detected traffic signs',
            'cross_evnt_city_0': 'Crossings without traffic lights',
            'cross_evnt_city_1': 'Crossings without traffic lights',
            'traffic_mortality_0': 'Traffic mortality',
            'traffic_mortality_1': 'Traffic mortality',
            'literacy_rate_0': 'Literacy rate',
            'literacy_rate_1': 'Literacy rate',
            'gini_0': 'Gini coefficient',
            'gini_1': 'Gini coefficient',
            'med_age_0': 'Median age',
            'med_age_1': 'Median age',
            }

        corr_matrix_day = corr_matrix_day.rename(columns=rename_dict_1, index=rename_dict_1)
        corr_matrix_night = corr_matrix_night.rename(columns=rename_dict_1, index=rename_dict_1)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_day,
                        text_auto=".2f",  # Display correlation values on the heatmap # type: ignore
                        color_continuous_scale='RdBu',  # Color scale
                        aspect="auto")  # Automatically adjust aspect ratio

        fig.update_layout(coloraxis_showscale=False)

        # Update font family and size
        fig.update_layout(
            width=1600,
            height=900,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),  # Remove all margins
            font=dict(
                family=common.get_configs('font_family'),
                size=common.get_configs('font_size')
            )
        )

        # Rotate Y axis labels (change angle as desired)
        fig.update_yaxes(tickangle=0, automargin=True)  # 90 for vertical, 45 for slanted

        # Set font size and family for annotation text
        fig.update_traces(
            textfont_size=18,
            textfont_family=common.get_configs('font_family')
            )

        plots_io_class.save_plotly_figure(fig, "correlation_matrix_heatmap_day", save_final=True)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_night, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation Matrix Heatmap in night"  # Title of the heatmap
                        )

        fig.update_layout(coloraxis_showscale=False)

        # Update font family and size
        fig.update_layout(
            width=1600,
            height=900,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),  # Remove all margins
            font=dict(
                family=common.get_configs('font_family'),
                size=common.get_configs('font_size')
            )
        )

        # Rotate Y axis labels (change angle as desired)
        fig.update_yaxes(tickangle=0, automargin=True)  # 90 for vertical, 45 for slanted

        # Set font size and family for annotation text
        fig.update_traces(
            textfont_size=18,
            textfont_family=common.get_configs('font_family')
            )

        plots_io_class.save_plotly_figure(fig, "correlation_matrix_heatmap_night", save_final=True)

        # Initialise a list to store rows of data (one row per country)
        data_rows = []

        # Assuming `conditions` is a list of conditions working with
        conditions = ['0', '1']  # Modify this list to include all conditions (e.g., '0', '1', etc.)

        # Iterate over each country and condition
        for country in final_dict:
            # Initialise a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'traffic_mortality', 'literacy_rate', 'gini', 'med_age']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[country].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[country].get("avg_day_night_speed", [])
                time_vals = final_dict[country].get("avg_day_night_time", [])

                if speed_vals:  # Avoid division by zero or empty arrays
                    row_data["avg_day_night_speed"] = np.mean(speed_vals)
                else:
                    row_data["avg_day_night_speed"] = np.nan  # Handle empty or missing arrays

                if time_vals:
                    row_data["avg_day_night_time"] = np.mean(time_vals)
                else:
                    row_data["avg_day_night_time"] = np.nan  # Handle empty or missing arrays

            # Append the row data for the current country
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)
        # df = df[[col for col in df.columns if col.endswith("_0") or col.endswith("_1")]]

        # Create a new DataFrame to average the columns across conditions
        agg_df = pd.DataFrame()

        # Define known conditions (from earlier)
        conditions = ['0', '1']

        for col in df.columns:
            # Check if the column ends with a known condition
            if any(col.endswith(f"_{cond}") for cond in conditions):
                feature_name = "_".join(col.split("_")[:-1])
                if feature_name not in agg_df.columns:
                    condition_cols = [c for c in df.columns if c.startswith(feature_name + "_")]
                    agg_df[feature_name] = df[condition_cols].mean(axis=1)
            else:
                # Directly copy columns that don't follow the condition pattern (like avg_day_night_speed)
                agg_df[col] = df[col]

            # Create a new column by averaging values across conditions for the same feature
            if feature_name not in agg_df.columns:
                # Select the columns for this feature across all conditions
                condition_cols = [c for c in df.columns if c.startswith(feature_name + "_")]  # type: ignore
                agg_df[feature_name] = df[condition_cols].mean(axis=1)

        ordered_features = ['avg_day_night_speed', 'avg_day_night_time',
                            'ped_cross_city', 'person_city', 'bicycle_city', 'car_city',
                            'motorcycle_city', 'bus_city', 'truck_city', 'vehicle_city',
                            'cellphone_city', 'trf_sign_city', 'cross_evnt_city',
                            'traffic_mortality', 'literacy_rate', 'gini', 'med_age']

        ordered_features_in_df = [col for col in ordered_features if col in agg_df.columns]
        agg_df = agg_df[ordered_features_in_df]
        # Compute the correlation matrix on the aggregated DataFrame
        corr_matrix_avg = agg_df.corr(method='spearman')  # type: ignore

        # Rename the variables in the correlation matrix (example: renaming keys)
        rename_dict_2 = {
            'avg_day_night_speed': 'Crossing speed',
            'avg_day_night_time': 'Crossing initiation time',
            'ped_cross_city': 'Detected crossings',
            'person_city': 'Detected persons',
            'bicycle_city': 'Detected bicycles',
            'car_city': 'Detected cars',
            'motorcycle_city': 'Detected motorcycles',
            'bus_city': 'Detected buses',
            'truck_city': 'Detected trucks',
            'vehicle_city': 'Detected all motor vehicles',
            'cellphone_city': 'Detected cellphones',
            'trf_sign_city': 'Detected traffic signs',
            'cross_evnt_city': 'Crossings without traffic light',
            'traffic_mortality': 'Traffic mortality',
            'literacy_rate': 'Literacy rate',
            'gini': 'Gini coefficient',
            'med_age': 'Median age'
            }

        corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_2, index=rename_dict_2)

        # Generate the heatmap using Plotly
        fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                        color_continuous_scale='RdBu',  # Color scale
                        aspect="auto",  # Automatically adjust aspect ratio
                        # title="Correlation matrix heatmap averaged" # Title of the heatmap
                        )
        fig.update_layout(coloraxis_showscale=False)

        # Update font family and size
        fig.update_layout(
            width=1600,
            height=900,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),  # Remove all margins
            font=dict(
                family=common.get_configs('font_family'),
                size=common.get_configs('font_size')
            )
        )

        fig.update_traces(textfont_size=common.get_configs('font_size'))
        fig.update_xaxes(tickangle=45, tickfont=dict(size=common.get_configs('font_size')))
        fig.update_yaxes(tickangle=0, tickfont=dict(size=common.get_configs('font_size')))

        # Set font size and family for annotation text
        fig.update_traces(
            textfont_size=18,
            textfont_family=common.get_configs('font_family')
            )

        plots_io_class.save_plotly_figure(fig, "correlation_matrix_heatmap_averaged", save_final=True)

        # Continent Wise

        # Initialise a list to store rows of data (one row per country)
        data_rows = []

        # Assuming `conditions` is a list of conditions working with
        conditions = ['0', '1']  # Modify this list to include all conditions (e.g., '0', '1', etc.)
        unique_continents = df_mapping['continent'].unique()

        # Iterate over each country and condition
        for country in final_dict:
            # Initialise a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_city', 'person_city', 'bicycle_city', 'car_city', 'motorcycle_city', 'bus_city',
                            'truck_city', 'cross_evnt_city', 'vehicle_city', 'cellphone_city', 'trf_sign_city',
                            'traffic_mortality', 'literacy_rate', 'continent', 'gini', 'med_age']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[country].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[country].get("avg_day_night_speed", [])
                time_vals = final_dict[country].get("avg_day_night_time", [])

                if speed_vals:  # Avoid division by zero or empty arrays
                    row_data["avg_day_night_speed"] = np.mean(speed_vals)
                else:
                    row_data["avg_day_night_speed"] = np.nan  # Handle empty or missing arrays

                if time_vals:
                    row_data["avg_day_night_time"] = np.mean(time_vals)
                else:
                    row_data["avg_day_night_time"] = np.nan  # Handle empty or missing arrays

            # Append the row data for the current country
            data_rows.append(row_data)

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data_rows)

        for continents in unique_continents:
            filtered_df = df[(df['continent_0'] == continents) | (df['continent_1'] == continents)]
            # Create a new DataFrame to average the columns across conditions
            agg_df = pd.DataFrame()

            # Define known conditions (from earlier)
            conditions = ['0', '1']

            for col in filtered_df.columns:
                # Check if the column ends with a known condition
                if any(col.endswith(f"_{cond}") for cond in conditions):
                    feature_name = "_".join(col.split("_")[:-1])
                    # Skip columns named "continent_0" or "continent_1"
                    if "continent" in feature_name:
                        continue
                    if feature_name not in agg_df.columns:
                        condition_cols = [c for c in filtered_df.columns if c.startswith(feature_name + "_")]
                        if all(pd.api.types.is_numeric_dtype(filtered_df[c]) for c in condition_cols):
                            agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)
                        else:
                            logger.debug(f"Skipping non-numeric feature: {feature_name}")

                else:
                    agg_df[col] = filtered_df[col]

                # Create a new column by averaging values across conditions for the same feature
                if feature_name not in agg_df.columns:
                    # Select the columns for this feature across all conditions
                    condition_cols = [c for c in filtered_df.columns if feature_name in c]
                    agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)

            ordered_features_in_df = [col for col in ordered_features if col in agg_df.columns]
            agg_df = agg_df[ordered_features_in_df]

            # Compute the correlation matrix on the aggregated DataFrame
            corr_matrix_avg = agg_df.corr(method='spearman')  # type: ignore

            # Rename the variables in the correlation matrix (example: renaming keys)
            rename_dict_3 = {
                'avg_day_night_speed': 'Crossing speed',
                "avg_day_night_time": 'Crossing initiation time',
                'ped_cross_city': 'Detected crossings',
                'person_city': 'Detected persons',
                'bicycle_city': 'Detected bicycles',
                'car_city': 'Detected cars',
                'motorcycle_city': 'Detected motorcycles',
                'bus_city': 'Detected buses',
                'truck_city': 'Detected trucks',
                'vehicle_city': 'Detected all motor vehicles',
                'cellphone_city': 'Detected cellphones',
                'trf_sign_city': 'Detected traffic signs',
                'cross_evnt_city': 'Crossings without traffic light',
                'traffic_mortality': 'Traffic mortality',
                'literacy_rate': 'Literacy rate',
                'gini': 'Gini coefficient', 'med_age': 'Median age'
                }

            corr_matrix_avg = corr_matrix_avg.rename(columns=rename_dict_3, index=rename_dict_3)

            # Generate the heatmap using Plotly
            fig = px.imshow(corr_matrix_avg, text_auto=".2f",  # Display correlation values on heatmap  # type: ignore
                            color_continuous_scale='RdBu',  # Color scale
                            aspect="auto",  # Automatically adjust aspect ratio
                            # title=f"Correlation matrix heatmap {continents}"  # Title of the heatmap
                            )

            fig.update_layout(coloraxis_showscale=False)

            fig.update_layout(
                coloraxis_showscale=False,
                width=1600,
                height=900,
                margin=dict(l=0, r=0, t=0, b=0),  # Remove all margins
                font=dict(
                    family=common.get_configs('font_family'),
                    size=common.get_configs('font_size')
                )
            )

            # Update text font size inside heatmap
            fig.update_traces(textfont_size=14)
            fig.update_xaxes(tickangle=45, tickfont=dict(size=18))
            fig.update_yaxes(tickangle=0, tickfont=dict(size=18))

            # Set font size and family for annotation text
            fig.update_traces(
                textfont_size=18,
                textfont_family=common.get_configs('font_family')
                )

            # save file to local output folder
            if save_file:
                # Final adjustments and display
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                plots_io_class.save_plotly_figure(fig, f"correlation_matrix_heatmap_{continents}", save_final=True)
            # open it in localhost instead
            else:
                fig.show()
