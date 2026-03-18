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

    def correlation_matrix(self, df_mapping, ped_cross_locality, person_locality, bicycle_locality, car_locality,
                           motorcycle_locality, bus_locality, truck_locality, cross_evnt_locality, vehicle_locality,
                           cellphone_locality, trf_sign_locality, speed_values, time_values, avg_time, avg_speed):
        """
        Compute and visualise correlation matrices for various locality-level traffic and demographic data.

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
            df_mapping (pd.DataFrame): A mapping DataFrame containing metadata for each locality-state combination,
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

        # Now populate the final_dict with locality-wise data
        for locality_condition, speed in avg_speed.items():
            locality, lat, long, condition = locality_condition.split('_')

            # Get the country from the previously stored locality_country_map
            country = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "country")
            iso_code = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "iso3")
            continent = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "continent")
            population_country = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "population_country")  # noqa: E501
            gdp_locality = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "gmp")
            traffic_mortality = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "traffic_mortality")  # noqa: E501
            literacy_rate = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "literacy_rate")  # noqa: E501
            gini = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "gini")
            traffic_index = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "traffic_index")  # noqa: E501

            if country or iso_code is not None:

                # Initialise the locality's dictionary if not already present
                if f'{locality}_{lat}_{long}' not in final_dict:
                    final_dict[f'{locality}_{lat}_{long}'] = {
                        "avg_speed_0": None, "avg_speed_1": None, "avg_time_0": None, "avg_time_1": None,
                        "speed_val_0": None, "speed_val_1": None, "time_val_0": None, "time_val_1": None,
                        "ped_cross_locality_0": 0, "ped_cross_locality_1": 0,
                        "person_locality_0": 0, "person_locality_1": 0, "bicycle_locality_0": 0,
                        "bicycle_locality_1": 0, "car_locality_0": 0, "car_locality_1": 0,
                        "motorcycle_locality_0": 0, "motorcycle_locality_1": 0, "bus_locality_0": 0,
                        "bus_locality_1": 0, "truck_locality_0": 0, "truck_locality_1": 0,
                        "cross_evnt_locality_0": 0, "cross_evnt_locality_1": 0, "vehicle_locality_0": 0,
                        "vehicle_locality_1": 0, "cellphone_locality_0": 0, "cellphone_locality_1": 0,
                        "trf_sign_locality_0": 0, "trf_sign_locality_1": 0,
                    }

                # Populate the corresponding speed and time based on the condition
                final_dict[f'{locality}_{lat}_{long}'][f"avg_speed_{condition}"] = speed
                if f'{locality}_{lat}_{long}_{condition}' in avg_time:
                    final_dict[f'{locality}_{lat}_{long}'][f"avg_time_{condition}"] = avg_time.get(
                        f'{locality}_{lat}_{long}_{condition}', None)
                    final_dict[f'{locality}_{lat}_{long}'][f"time_val_{condition}"] = time_values.get(
                        f'{locality}_{lat}_{long}_{condition}', None)
                    final_dict[f'{locality}_{lat}_{long}'][f"speed_val_{condition}"] = speed_values.get(
                        f'{locality}_{lat}_{long}_{condition}', None)
                    final_dict[f'{locality}_{lat}_{long}'][f"time_val_{condition}"] = time_values.get(
                        f'{locality}_{lat}_{long}_{condition}', None)
                    final_dict[f'{locality}_{lat}_{long}'][f"ped_cross_locality_{condition}"] = ped_cross_locality.get(
                        f'{locality}_{lat}_{long}_{condition}', None)

                    avg_person_locality = tools_class.compute_avg_variable_locality(person_locality)
                    final_dict[f'{locality}_{lat}_{long}'][f"person_locality_{condition}"] = avg_person_locality.get(
                        f'{locality}_{lat}_{long}_{condition}', 0)

                    avg_bicycle_locality = tools_class.compute_avg_variable_locality(bicycle_locality)
                    final_dict[f'{locality}_{lat}_{long}'][f"bicycle_locality_{condition}"] = avg_bicycle_locality.get(
                        f'{locality}_{lat}_{long}_{condition}', 0)

                    avg_car_locality = tools_class.compute_avg_variable_locality(car_locality)
                    final_dict[f'{locality}_{lat}_{long}'][f"car_locality_{condition}"] = avg_car_locality.get(
                        f'{locality}_{lat}_{long}_{condition}', 0)

                    avg_motorcycle_locality = tools_class.compute_avg_variable_locality(motorcycle_locality)
                    final_dict[f'{locality}_{lat}_{long}'][f"motorcycle_locality_{condition}"] = avg_motorcycle_locality.get(  # noqa: E501
                        f'{locality}_{lat}_{long}_{condition}', 0)

                    avg_bus_locality = tools_class.compute_avg_variable_locality(bus_locality)
                    final_dict[f'{locality}_{lat}_{long}'][f"bus_locality_{condition}"] = avg_bus_locality.get(
                        f'{locality}_{lat}_{long}_{condition}', 0)

                    avg_truck_locality = tools_class.compute_avg_variable_locality(truck_locality)
                    final_dict[f'{locality}_{lat}_{long}'][f"truck_locality_{condition}"] = avg_truck_locality.get(
                        f'{locality}_{lat}_{long}_{condition}', 0)

                    final_dict[f'{locality}_{lat}_{long}'][f"cross_evnt_locality_{condition}"] = cross_evnt_locality.get(  # noqa: E501
                        f'{locality}_{lat}_{long}_{condition}', 0)

                    avg_vehicle_locality = tools_class.compute_avg_variable_locality(vehicle_locality)
                    final_dict[f'{locality}_{lat}_{long}'][f"vehicle_locality_{condition}"] = avg_vehicle_locality.get(
                        f'{locality}_{lat}_{long}_{condition}', None)

                    avg_cellphone_locality = tools_class.compute_avg_variable_locality(cellphone_locality)
                    final_dict[f'{locality}_{lat}_{long}'][f"cellphone_locality_{condition}"] = avg_cellphone_locality.get(  # noqa: E501
                        f'{locality}_{lat}_{long}_{condition}', 0)

                    avg_trf_sign_locality = tools_class.compute_avg_variable_locality(trf_sign_locality)
                    final_dict[f'{locality}_{lat}_{long}'][f"trf_sign_locality_{condition}"] = avg_trf_sign_locality.get(  # noqa: E501
                        f'{locality}_{lat}_{long}_{condition}', 0)

                    final_dict[f'{locality}_{lat}_{long}'][f"traffic_mortality_{condition}"] = traffic_mortality
                    final_dict[f'{locality}_{lat}_{long}'][f"literacy_rate_{condition}"] = literacy_rate
                    final_dict[f'{locality}_{lat}_{long}'][f"gini_{condition}"] = gini
                    final_dict[f'{locality}_{lat}_{long}'][f"traffic_index_{condition}"] = traffic_index
                    final_dict[f'{locality}_{lat}_{long}'][f"continent_{condition}"] = continent
                    if gdp_locality is not None:
                        final_dict[f'{locality}_{lat}_{long}'][f"gmp_{condition}"] = gdp_locality/population_country

        # Initialise an empty list to store the rows for the DataFrame
        data_day, data_night = [], []

        # Loop over each locality and gather relevant values for condition 0
        for locality in final_dict:
            # Initialise a dictionary for the row
            row_day, row_night = {}, {}

            # Add data for condition 0 (ignore 'speed_val' and 'time_val')
            for condition in ['0']:  # Only include condition 0
                for key, value in final_dict[locality].items():
                    if condition in key and 'speed_val' not in key and 'time_val' not in key and 'continent' not in key:  # noqa:E501
                        row_day[key] = value

            # Append the row to the data list
            data_day.append(row_day)

            for condition in ['1']:  # Only include condition 1
                for key, value in final_dict[locality].items():
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
            'ped_cross_locality_0': 'Crossing', 'ped_cross_locality_1': 'Crossing',
            'person_locality_0': 'Detected persons', 'person_locality_1': 'Detected persons',
            'bicycle_locality_0': 'Detected bicycles', 'bicycle_locality_1': 'Detected bicycles',
            'car_locality_0': 'Detected cars', 'car_locality_1': 'Detected cars',
            'motorcycle_locality_0': 'Detected motorcycles', 'motorcycle_locality_1': 'Detected motorcycles',
            'bus_locality_0': 'Detected buses', 'bus_locality_1': 'Detected buses',
            'truck_locality_0': 'Detected trucks', 'truck_locality_1': 'Detected trucks',
            'cross_evnt_locality_0': 'Crossings without traffic lights',
            'cross_evnt_locality_1': 'Crossings without traffic lights',
            'vehicle_locality_0': 'Detected motor vehicles',
            'vehicle_locality_1': 'Detected motor vehicles',
            'cellphone_locality_0': 'Detected cellphones', 'cellphone_locality_1': 'Detected cellphones',
            'trf_sign_locality_0': 'Detected traffic signs', 'trf_sign_locality_1': 'Detected traffic signs',
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

        # Initialise a list to store rows of data (one row per locality)
        data_rows = []

        # Assuming `conditions` is a list of conditions working with
        conditions = ['0', '1']  # Modify this list to include all condition (e.g., '0', '1', etc.)

        # Iterate over each locality and condition
        for locality in final_dict:
            # Initialise a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_locality', 'person_locality', 'bicycle_locality', 'car_locality',
                            'motorcycle_locality', 'bus_locality', 'truck_locality', 'cross_evnt_locality',
                            'vehicle_locality', 'cellphone_locality', 'trf_sign_locality',
                            'gmp', 'traffic_mortality', 'literacy_rate', 'gini', 'traffic_index']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[locality].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[locality].get(f"speed_val_{condition}", [])
                time_vals = final_dict[locality].get(f"time_val_{condition}", [])

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

            # Append the row data for the current locality
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
            'ped_cross_locality': 'Crossing', 'person_locality': 'Detected persons',
            'bicycle_locality': 'Detected bicycles', 'car_locality': 'Detected cars',
            'motorcycle_locality': 'Detected motorcycles', 'bus_locality': 'Detected buses',
            'truck_locality': 'Detected trucks', 'cross_evnt_locality': 'Crossings without traffic light',
            'vehicle_locality': 'Detected all motor vehicles', 'cellphone_locality': 'Detected cellphones',
            'trf_sign_locality': 'Detected traffic signs', 'gmp_locality': 'GMP',
            'traffic_mortality_locality': 'Traffic mortality', 'literacy_rate_locality': 'Literacy rate',
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

        # Initialise a list to store rows of data (one row per locality)
        data_rows = []

        # Assuming `conditions` is a list of conditions working with
        conditions = ['0', '1']  # Modify this list to include all conditions (e.g., '0', '1', etc.)
        unique_continents = df_mapping['continent'].unique()

        # Iterate over each locality and condition
        for locality in final_dict:
            # Initialise a dictionary to store the values for the current row
            row_data = {}

            # For each condition
            for condition in conditions:
                # For each variable (exclude avg_speed and avg_time)
                for var in ['ped_cross_locality', 'person_locality', 'bicycle_locality', 'car_locality',
                            'motorcycle_locality', 'bus_locality', 'truck_locality', 'cross_evnt_locality',
                            'vehicle_locality', 'cellphone_locality', 'trf_sign_locality',
                            'gmp', 'traffic_mortality', 'literacy_rate', 'continent', 'gini', 'traffic_index']:
                    # Populate each variable in the row_data dictionary
                    row_data[f"{var}_{condition}"] = final_dict[locality].get(f"{var}_{condition}", 0)

                # Calculate average of speed_val and time_val (assumed to be arrays)
                speed_vals = final_dict[locality].get(f"speed_val_{condition}", [])
                time_vals = final_dict[locality].get(f"time_val_{condition}", [])

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

            # Append the row data for the current locality
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
                    agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)  # type: ignore

            # Compute the correlation matrix on the aggregated DataFrame
            corr_matrix_avg = agg_df.corr(method='spearman')

            # Rename the variables in the correlation matrix (example: renaming keys)
            rename_dict_3 = {
                'avg_speed_val': 'Crossing speed', 'avg_time_val': 'Crossing initiation time',
                'ped_cross_locality': 'Crossing', 'person_locality': 'Detected persons',
                'bicycle_locality': 'Detected bicycles', 'car_locality': 'Detected cars',
                'motorcycle_locality': 'Detected motorcycles', 'bus_locality': 'Detected buses',
                'truck_locality': 'Detected trucks', 'cross_evnt_locality': 'Crossings without traffic light',
                'vehicle_locality': 'Detected all motor vehicles', 'cellphone_locality': 'Detected cellphones',
                'trf_sign_locality': 'Detected traffic signs', 'gmp': 'GMP',
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

    def correlation_matrix_country(self, df_mapping, df_countries, ped_cross_locality, person_locality,
                                   bicycle_locality, car_locality, motorcycle_locality, bus_locality, truck_locality,
                                   cross_evnt_locality, vehicle_locality, cellphone_locality, trf_sign_locality,
                                   avg_speed_country, avg_time_country, cross_no_equip_country, save_file=True):

        """Generates and saves correlation matrices of traffic and demographic data by country.

        This method:
            1. Aggregates raw detection data from locality level to country level.
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
            ped_cross_locality (dict): Counts of pedestrian crossings by locality.
            person_locality (dict): Counts of detected persons by locality.
            bicycle_locality (dict): Counts of detected bicycles by locality.
            car_locality (dict): Counts of detected cars by locality.
            motorcycle_locality (dict): Counts of detected motorcycles by locality.
            bus_locality (dict): Counts of detected buses by locality.
            truck_locality (dict): Counts of detected trucks by locality.
            cross_evnt_locality (dict): Counts of crossing events without traffic lights by locality.
            vehicle_locality (dict): Counts of all motor vehicles by locality.
            cellphone_locality (dict): Counts of detected cellphones by locality.
            trf_sign_locality (dict): Counts of detected traffic signs by locality.
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

        # Aggregate locality-level counts to country level
        ped_cross_locality = grouping_class.country_sum_from_cities(ped_cross_locality, df_mapping)
        person_locality = grouping_class.country_averages_from_nested(person_locality, df_mapping)
        bicycle_locality = grouping_class.country_averages_from_nested(bicycle_locality, df_mapping)
        car_locality = grouping_class.country_averages_from_nested(car_locality, df_mapping)
        motorcycle_locality = grouping_class.country_averages_from_nested(motorcycle_locality, df_mapping)
        bus_locality = grouping_class.country_averages_from_nested(bus_locality, df_mapping)
        truck_locality = grouping_class.country_averages_from_nested(truck_locality, df_mapping)
        vehicle_locality = grouping_class.country_averages_from_nested(vehicle_locality, df_mapping)
        cellphone_locality = grouping_class.country_averages_from_nested(cellphone_locality, df_mapping)
        trf_sign_locality = grouping_class.country_averages_from_nested(trf_sign_locality, df_mapping)
        cross_evnt_locality = grouping_class.country_averages_from_flat(cross_evnt_locality, df_mapping)

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

                # Initialise the locality's dictionary if not already present
                if f'{country}' not in final_dict:
                    final_dict[f'{country}'] = {
                                                "avg_speed_0": None,
                                                "avg_speed_1": None,
                                                "avg_time_0": None,
                                                "avg_time_1": None,
                                                "avg_day_night_speed": None,
                                                "avg_day_night_time": None,
                                                "ped_cross_locality_0": 0,
                                                "ped_cross_locality_1": 0,
                                                "person_locality_0": 0,
                                                "person_locality_1": 0,
                                                "bicycle_locality_0": 0,
                                                "bicycle_locality_1": 0,
                                                "car_locality_0": 0,
                                                "car_locality_1": 0,
                                                "motorcycle_locality_0": 0,
                                                "motorcycle_locality_1": 0,
                                                "bus_locality_0": 0,
                                                "bus_locality_1": 0,
                                                "truck_locality_0": 0,
                                                "truck_locality_1": 0,
                                                "vehicle_locality_0": 0,
                                                "vehicle_locality_1": 0,
                                                "cellphone_locality_0": 0,
                                                "cellphone_locality_1": 0,
                                                "trf_sign_locality_0": 0,
                                                "trf_sign_locality_1": 0,
                                                "cross_evnt_locality_0": 0,
                                                "cross_evnt_locality_1": 0,
                                                }

                # Populate the corresponding speed and time based on the condition
                final_dict[f'{country}'][f"avg_speed_{condition}"] = speed

                if f'{country}_{condition}' in avg_time_country:
                    final_dict[f'{country}'][f"avg_time_{condition}"] = avg_time_country.get(
                        f'{country}_{condition}', None)
                    final_dict[f'{country}'][f"ped_cross_locality_{condition}"] = ped_cross_locality.get(
                        f'{country}_{condition}', None)
                    final_dict[f'{country}'][f"person_locality_{condition}"] = person_locality.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"bicycle_locality_{condition}"] = bicycle_locality.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"car_locality_{condition}"] = car_locality.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"motorcycle_locality_{condition}"] = motorcycle_locality.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"bus_locality_{condition}"] = bus_locality.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"truck_locality_{condition}"] = truck_locality.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"vehicle_locality_{condition}"] = vehicle_locality.get(
                        f'{country}_{condition}', None)
                    final_dict[f'{country}'][f"cellphone_locality_{condition}"] = cellphone_locality.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"trf_sign_locality_{condition}"] = trf_sign_locality.get(
                        f'{country}_{condition}', 0)
                    final_dict[f'{country}'][f"cross_evnt_locality_{condition}"] = cross_no_equip_country.get(
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

        # Loop over each locality and gather relevant values for condition 0
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
            'ped_cross_locality_0': 'Detected crossings',
            'ped_cross_locality_1': 'Detected crossings',
            'person_locality_0': 'Detected persons',
            'person_locality_1': 'Detected persons',
            'bicycle_locality_0': 'Detected bicycles',
            'bicycle_locality_1': 'Detected bicycles',
            'car_locality_0': 'Detected cars',
            'car_locality_1': 'Detected cars',
            'motorcycle_locality_0': 'Detected motorcycles',
            'motorcycle_locality_1': 'Detected motorcycles',
            'bus_locality_0': 'Detected buses',
            'bus_locality_1': 'Detected buses',
            'truck_locality_0': 'Detected trucks',
            'truck_locality_1': 'Detected trucks',
            'vehicle_locality_0': 'Detected all motor vehicles',
            'vehicle_locality_1': 'Detected all motor vehicles',
            'cellphone_locality_0': 'Detected cellphones',
            'cellphone_locality_1': 'Detected cellphones',
            'trf_sign_locality_0': 'Detected traffic signs',
            'trf_sign_locality_1': 'Detected traffic signs',
            'cross_evnt_locality_0': 'Crossings without traffic lights',
            'cross_evnt_locality_1': 'Crossings without traffic lights',
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
                for var in ['ped_cross_locality', 'person_locality', 'bicycle_locality', 'car_locality',
                            'motorcycle_locality', 'bus_locality', 'truck_locality', 'cross_evnt_locality',
                            'vehicle_locality', 'cellphone_locality', 'trf_sign_locality', 'traffic_mortality',
                            'literacy_rate', 'gini', 'med_age']:
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
                            'ped_cross_locality', 'person_locality', 'bicycle_locality', 'car_locality',
                            'motorcycle_locality', 'bus_locality', 'truck_locality', 'vehicle_locality',
                            'cellphone_locality', 'trf_sign_locality', 'cross_evnt_locality',
                            'traffic_mortality', 'literacy_rate', 'gini', 'med_age']

        ordered_features_in_df = [col for col in ordered_features if col in agg_df.columns]
        agg_df = agg_df[ordered_features_in_df]
        # Compute the correlation matrix on the aggregated DataFrame
        corr_matrix_avg = agg_df.corr(method='spearman')  # type: ignore

        # Rename the variables in the correlation matrix (example: renaming keys)
        rename_dict_2 = {
            'avg_day_night_speed': 'Crossing speed',
            'avg_day_night_time': 'Crossing initiation time',
            'ped_cross_locality': 'Detected crossings',
            'person_locality': 'Detected persons',
            'bicycle_locality': 'Detected bicycles',
            'car_locality': 'Detected cars',
            'motorcycle_locality': 'Detected motorcycles',
            'bus_locality': 'Detected buses',
            'truck_locality': 'Detected trucks',
            'vehicle_locality': 'Detected all motor vehicles',
            'cellphone_locality': 'Detected cellphones',
            'trf_sign_locality': 'Detected traffic signs',
            'cross_evnt_locality': 'Crossings without traffic light',
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
                for var in ['ped_cross_locality', 'person_locality', 'bicycle_locality', 'car_locality',
                            'motorcycle_locality', 'bus_locality', 'truck_locality', 'cross_evnt_locality',
                            'vehicle_locality', 'cellphone_locality', 'trf_sign_locality', 'traffic_mortality',
                            'literacy_rate', 'continent', 'gini', 'med_age']:
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
                            agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)  # type: ignore
                        else:
                            logger.debug(f"Skipping non-numeric feature: {feature_name}")

                else:
                    agg_df[col] = filtered_df[col]

                # Create a new column by averaging values across conditions for the same feature
                if feature_name not in agg_df.columns:
                    # Select the columns for this feature across all conditions
                    condition_cols = [c for c in filtered_df.columns if feature_name in c]
                    agg_df[feature_name] = filtered_df[condition_cols].mean(axis=1)  # type: ignore

            ordered_features_in_df = [col for col in ordered_features if col in agg_df.columns]
            agg_df = agg_df[ordered_features_in_df]

            # Compute the correlation matrix on the aggregated DataFrame
            corr_matrix_avg = agg_df.corr(method='spearman')  # type: ignore

            # Rename the variables in the correlation matrix (example: renaming keys)
            rename_dict_3 = {
                'avg_day_night_speed': 'Crossing speed',
                "avg_day_night_time": 'Crossing initiation time',
                'ped_cross_locality': 'Detected crossings',
                'person_locality': 'Detected persons',
                'bicycle_locality': 'Detected bicycles',
                'car_locality': 'Detected cars',
                'motorcycle_locality': 'Detected motorcycles',
                'bus_locality': 'Detected buses',
                'truck_locality': 'Detected trucks',
                'vehicle_locality': 'Detected all motor vehicles',
                'cellphone_locality': 'Detected cellphones',
                'trf_sign_locality': 'Detected traffic signs',
                'cross_evnt_locality': 'Crossings without traffic light',
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
