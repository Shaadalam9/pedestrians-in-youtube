import pickle
import statistics
import common
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from custom_logger import CustomLogger
from utils.plotting.io import IO
from utils.core.grouping import Grouping
from utils.core.metadata import MetaData
from utils.core.tools import Tools
from utils.plotting.layout import Layout
from utils.core.iso import ISO
from utils.plotting import constants as C
from utils.core.dataset_stats import Dataset_Stats

layout_class = Layout()
metadata_class = MetaData()
plots_io_class = IO()
grouping_class = Grouping()
tools_class = Tools()
iso_class = ISO()
dataset_stats = Dataset_Stats()

logger = CustomLogger(__name__)  # use custom logger

# File to store the city coordinates
file_results = 'results.pickle'


class Crossings:
    def __init__(self) -> None:
        pass

    def speed_and_time_to_start_cross(self, df_mapping, font_size_captions=40, x_axis_title_height=150, legend_x=0.81,
                                      legend_y=0.98, legend_spacing=0.02):
        logger.info("Plotting speed_and_time_to_start_cross")
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[25]
        avg_time = data_tuple[24]

        # Check if both 'speed' and 'time' are valid dictionaries
        if avg_speed is None or avg_time is None:
            raise ValueError("Either 'speed' or 'time' returned None, please check the input data or calculations.")

        # Remove the ones where there is data missing for a specific country and condition
        common_keys = avg_speed.keys() & avg_time.keys()

        # Retain only the key-value pairs where the key is present in both dictionaries
        avg_speed = {key: avg_speed[key] for key in common_keys}
        avg_time = {key: avg_time[key] for key in common_keys}

        # Now populate the final_dict with city-wise data
        for city_condition, speed in tqdm(avg_speed.items()):
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:

                # Initialise the city's dictionary if not already present
                if f'{city}_{lat}_{long}' not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {
                        "speed_0": None, "speed_1": None, "time_0": None, "time_1": None,
                        "country": country, "iso": iso_code}

                # Populate the corresponding speed and time based on the condition
                final_dict[f"{city}_{lat}_{long}"][f"speed_{condition}"] = speed
                if f'{city}_{lat}_{long}_{condition}' in avg_time:
                    final_dict[f"{city}_{lat}_{long}"][f"time_{condition}"] = avg_time[f'{city}_{lat}_{long}_{condition}']  # noqa: E501

        # Extract all valid speed_0 and speed_1 values along with their corresponding cities
        diff_speed_values = [(f'{city}', abs(data['speed_0'] - data['speed_1']))
                             for city, data in final_dict.items()
                             if data['speed_0'] is not None and data['speed_1'] is not None]

        if diff_speed_values:
            # Sort the list by the absolute difference and get the top 5 and bottom 5
            sorted_diff_speed_values = sorted(diff_speed_values, key=lambda x: x[1], reverse=True)

            top_5_max_speed = sorted_diff_speed_values[:5]  # Top 5 maximum differences
            top_5_min_speed = sorted_diff_speed_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("Top 5 cities with max |speed at day - speed at night| differences:")
            for city, diff in top_5_max_speed:
                city_state = grouping_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")

            logger.info("Top 5 cities with min |speed at day - speed at night| differences:")
            for city, diff in top_5_min_speed:
                city_state = grouping_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")
        else:
            logger.info("No valid speed_0 and speed_1 values found for comparison.")

        # Extract all valid time_0 and time_1 values along with their corresponding cities
        diff_time_values = [(city, abs(data['time_0'] - data['time_1']))
                            for city, data in final_dict.items()
                            if data['time_0'] is not None and data['time_1'] is not None]

        if diff_time_values:
            sorted_diff_time_values = sorted(diff_time_values, key=lambda x: x[1], reverse=True)

            top_5_max = sorted_diff_time_values[:5]  # Top 5 maximum differences
            top_5_min = sorted_diff_time_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("Top 5 cities with max |time_0 - time_1| differences:")
            for city, diff in top_5_max:
                city_state = grouping_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")

            logger.info("Top 5 cities with min |time_0 - time_1| differences:")
            for city, diff in top_5_min:
                city_state = grouping_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")
        else:
            logger.info("No valid time_0 and time_1 values found for comparison.")

        # Filtering out entries where entries is None
        filtered_dict_s_0 = {city: info for city, info in final_dict.items() if info["speed_0"] is not None}
        filtered_dict_s_1 = {city: info for city, info in final_dict.items() if info["speed_1"] is not None}
        filtered_dict_t_0 = {city: info for city, info in final_dict.items() if info["time_0"] is not None}
        filtered_dict_t_1 = {city: info for city, info in final_dict.items() if info["time_1"] is not None}

        # Find city with max and min speed_0 and speed_1
        if filtered_dict_s_0:
            max_speed_city_0 = max(filtered_dict_s_0, key=lambda city: filtered_dict_s_0[city]["speed_0"])
            min_speed_city_0 = min(filtered_dict_s_0, key=lambda city: filtered_dict_s_0[city]["speed_0"])
            max_speed_value_0 = filtered_dict_s_0[max_speed_city_0]["speed_0"]
            min_speed_value_0 = filtered_dict_s_0[min_speed_city_0]["speed_0"]

            logger.info(f"City with max speed at day: {grouping_class.process_city_string(max_speed_city_0, df_mapping)} with speed of {max_speed_value_0} m/s")  # noqa:E501
            logger.info(f"City with min speed at day: {grouping_class.process_city_string(min_speed_city_0, df_mapping)} with speed of {min_speed_value_0} m/s")  # noqa:E501

        if filtered_dict_s_1:
            max_speed_city_1 = max(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            min_speed_city_1 = min(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            max_speed_value_1 = filtered_dict_s_1[max_speed_city_1]["speed_1"]
            min_speed_value_1 = filtered_dict_s_1[min_speed_city_1]["speed_1"]

            logger.info(f"City with max speed at night: {grouping_class.process_city_string(max_speed_city_1, df_mapping)} with speed of {max_speed_value_1} m/s")  # noqa:E501
            logger.info(f"City with min speed at night: {grouping_class.process_city_string(min_speed_city_1, df_mapping)} with speed of {min_speed_value_1} m/s")  # noqa:E501

        # Find city with max and min time_0 and time_1
        if filtered_dict_t_0:
            max_time_city_0 = max(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            min_time_city_0 = min(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            max_time_value_0 = filtered_dict_t_0[max_time_city_0]["time_0"]
            min_time_value_0 = filtered_dict_t_0[min_time_city_0]["time_0"]

            logger.info(f"City with max time at day: {grouping_class.process_city_string(max_time_city_0, df_mapping)} with time of {max_time_value_0} s")  # noqa:E501
            logger.info(f"City with min time at day: {grouping_class.process_city_string(min_time_city_0, df_mapping)} with time of {min_time_value_0} s")  # noqa:E501

        if filtered_dict_t_1:
            max_time_city_1 = max(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            min_time_city_1 = min(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            max_time_value_1 = filtered_dict_t_1[max_time_city_1]["time_1"]
            min_time_value_1 = filtered_dict_t_1[min_time_city_1]["time_1"]

            logger.info(f"City with max time at night: {grouping_class.process_city_string(max_time_city_1, df_mapping)} with time of {max_time_value_1} s")  # noqa:E501
            logger.info(f"City with min time at night: {grouping_class.process_city_string(min_time_city_1, df_mapping)} with time of {min_time_value_1} s")  # noqa:E501

        # Extract valid speed and time values and calculate statistics
        speed_0_values = [data['speed_0'] for data in final_dict.values() if pd.notna(data['speed_0'])]
        speed_1_values = [data['speed_1'] for data in final_dict.values() if pd.notna(data['speed_1'])]
        time_0_values = [data['time_0'] for data in final_dict.values() if pd.notna(data['time_0'])]
        time_1_values = [data['time_1'] for data in final_dict.values() if pd.notna(data['time_1'])]

        if speed_0_values:
            mean_speed_0 = statistics.mean(speed_0_values)
            sd_speed_0 = statistics.stdev(speed_0_values) if len(speed_0_values) > 1 else 0
            logger.info(f"Mean of speed during day time: {mean_speed_0}")
            logger.info(f"Standard deviation of speed during day time: {sd_speed_0}")
        else:
            logger.error("No valid speed during day time values found.")

        if speed_1_values:
            mean_speed_1 = statistics.mean(speed_1_values)
            sd_speed_1 = statistics.stdev(speed_1_values) if len(speed_1_values) > 1 else 0
            logger.info(f"Mean of speed during night time: {mean_speed_1}")
            logger.info(f"Standard deviation of speed during night time: {sd_speed_1}")
        else:
            logger.error("No valid speed during night time values found.")

        if time_0_values:
            mean_time_0 = statistics.mean(time_0_values)
            sd_time_0 = statistics.stdev(time_0_values) if len(time_0_values) > 1 else 0
            logger.info(f"Mean of time during day time: {mean_time_0}")
            logger.info(f"Standard deviation of time during day time: {sd_time_0}")
        else:
            logger.error("No valid time during day time values found.")

        if time_1_values:
            mean_time_1 = statistics.mean(time_1_values)
            sd_time_1 = statistics.stdev(time_1_values) if len(time_1_values) > 1 else 0
            logger.info(f"Mean of time during night time: {mean_time_1}")
            logger.info(f"Standard deviation of time during night time: {sd_time_1}")
        else:
            logger.error("No valid time during night time values found.")

        # Extract city, condition, and count_ from the info dictionary
        cities, conditions_, counts = [], [], []
        for key, value in tqdm(avg_time.items()):
            city, lat, long, condition = key.split('_')
            cities.append(f'{city}_{lat}_{long}')
            conditions_.append(condition)
            counts.append(value)

        # Combine keys from speed and time to ensure we include all available cities and conditions
        all_keys = set(avg_speed.keys()).union(set(avg_time.keys()))

        # Extract unique cities
        cities = list(set(["_".join(key.split('_')[:2]) for key in all_keys]))

        country_city_map = {}
        for city_state, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city_state)

        # Flatten the city list based on country groupings
        cities_ordered = []
        for country in sorted(country_city_map.keys()):  # Sort countries alphabetically
            cities_in_country = sorted(country_city_map[country])  # Sort cities within each country alphabetically
            cities_ordered.extend(cities_in_country)

        # Prepare data for day and night stacking
        day_avg_speed = [final_dict[city]['speed_0'] for city in cities_ordered]
        night_avg_speed = [final_dict[city]['speed_1'] for city in cities_ordered]
        day_time_dict = [final_dict[city]['time_0'] for city in cities_ordered]
        night_time_dict = [final_dict[city]['time_1'] for city in cities_ordered]

        # Ensure that plotting uses cities_ordered
        assert len(cities_ordered) == len(day_avg_speed) == len(night_avg_speed) == len(
            day_time_dict) == len(night_time_dict), "Lengths of lists don't match!"

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups

        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * C.BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col * 2, cols=2,  # Two columns
            vertical_spacing=0,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[2.0] * (num_cities_per_col * 2),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city = grouping_class.process_city_string(city, df_mapping)

            # Row for speed (Day and Night)
            row = 2 * i + 1
            if day_avg_speed[i] is not None and night_avg_speed[i] is not None:
                value = (day_avg_speed[i] + night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night",
                    marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_avg_speed[i] is not None:  # Only day data available
                value = (day_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_avg_speed[i] is not None:  # Only night data available
                value = (night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night",
                    marker=dict(color=C.BAR_COLOR_2), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            # Row for time (Day and Night)
            row = 2 * i + 2
            if day_time_dict[i] is not None and night_time_dict[i] is not None:
                value = (day_time_dict[i] + night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=C.BAR_COLOR_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=C.BAR_COLOR_4), text=[''],
                    textposition='auto', showlegend=False), row=row, col=1)

            elif day_time_dict[i] is not None:  # Only day time data available
                value = (day_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=C.BAR_COLOR_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_time_dict[i] is not None:  # Only night time data available
                value = (night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=C.BAR_COLOR_4),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

        # Similarly for the right column
        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city = grouping_class.process_city_string(city, df_mapping)

            row = 2 * i + 1
            idx = num_cities_per_col + i
            if day_avg_speed[idx] is not None and night_avg_speed[idx] is not None:
                value = (day_avg_speed[idx] + night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_avg_speed[idx] is not None:
                value = (day_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_avg_speed[idx] is not None:
                value = (night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            row = 2 * i + 2
            if day_time_dict[idx] is not None and night_time_dict[idx] is not None:
                value = (day_time_dict[idx] + night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=C.BAR_COLOR_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=C.BAR_COLOR_4), text=[''],
                    textposition='inside', showlegend=False), row=row, col=2)

            elif day_time_dict[idx] is not None:  # Only day time data available
                value = (day_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=C.BAR_COLOR_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_time_dict[idx] is not None:  # Only night time data available
                value = (night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=C.BAR_COLOR_4),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_time = max([
            (day_time_dict[i] if day_time_dict[i] is not None else 0) +
            (night_time_dict[i] if night_time_dict[i] is not None else 0)
            for i in range(len(cities_ordered))
        ]) if cities_ordered else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=False
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=False
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(
            title=dict(text="Mean speed of crossing (in m/s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=1
        )
        fig.update_xaxes(
            title=dict(text="Mean speed of crossing (in m/s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=2
        )
        fig.update_xaxes(
            title=dict(text="Mean time to start crossing (in s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=num_cities_per_col * 2,
            col=1
        )

        fig.update_xaxes(
            title=dict(text="Mean time to start crossing (in s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=num_cities_per_col * 2,
            col=2
        )

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=TALL_FIG_HEIGHT*2, width=4960, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Set the x-axis range to cover the values you want in x_grid_values
        x_grid_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x', yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x2', yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Define the legend items
        legend_items = [
            {"name": "Crossing speed during daytime", "color": C.BAR_COLOR_1},
            {"name": "Crossing speed during night time", "color": C.BAR_COLOR_2},
            {"name": "Crossing decision time during daytime", "color": C.BAR_COLOR_3},
            {"name": "Crossing decision time during night time", "color": C.BAR_COLOR_4},
        ]

        # Add the vertical legends at the top and bottom
        layout_class.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
                                                     spacing=legend_spacing, font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=1, x1=0.495, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.505, y0=1, x1=1, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Split cities into left and right columns
        left_column_cities = cities_ordered[:num_cities_per_col]
        right_column_cities = cities_ordered[num_cities_per_col:]

        # Adjust x positioning for the left and right columns
        x_position_left = 0.0  # Position for the left column
        x_position_right = 1.0  # Position for the right column
        font_size = 15  # Font size for visibility

        # Initialise variables for dynamic y positioning for both columns
        current_row_left = 1  # Start from the first row for the left column
        current_row_right = 1  # Start from the first row for the right column
        y_position_map_left = {}  # Store y positions for each country (left column)
        y_position_map_right = {}  # Store y positions for each country (right column)

        # Calculate the y positions dynamically for the left column
        for city in left_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_left:  # Add the country label once per country
                y_position_map_left[country] = 1 - (current_row_left - 1) / ((len(left_column_cities)-0.56) * 2)

            current_row_left += 2  # Increment the row for each city (speed and time take two rows)

        # Calculate the y positions dynamically for the right column
        for city in right_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_right:  # Add the country label once per country
                y_position_map_right[country] = 1 - (current_row_right - 1) / ((len(right_column_cities)-0.56) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        # Add annotations for country names dynamically for the left column
        for country, y_position in y_position_map_left.items():
            iso2 = iso_class.iso3_to_iso2(country)
            country = country + iso_class.iso2_to_flag(iso2)
            fig.add_annotation(
                x=x_position_left,  # Left column x position
                y=y_position,  # Calculated y position based on the city order
                xref="paper", yref="paper",
                text=country,  # Country name
                showarrow=False,
                font=dict(size=font_size, color="black"),
                xanchor='right',
                align='right',
                bgcolor='rgba(255,255,255,0.8)',  # Background color for visibility
                # bordercolor="black",  # Border for visibility
            )

        # Add annotations for country names dynamically for the right column
        for country, y_position in y_position_map_right.items():
            iso2 = iso_class.iso3_to_iso2(country)
            country = country + iso_class.iso2_to_flag(iso2)
            fig.add_annotation(
                x=x_position_right,  # Right column x position
                y=y_position,  # Calculated y position based on the city order
                xref="paper", yref="paper",
                text=country,  # Country name
                showarrow=False,
                font=dict(size=font_size, color="black"),
                xanchor='left',
                align='left',
                bgcolor='rgba(255,255,255,0.8)',  # Background color for visibility
                # bordercolor="black",  # Border for visibility
            )

        fig.update_yaxes(
            tickfont=dict(size=14, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )

        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=80, t=x_axis_title_height, b=x_axis_title_height))

        plots_io_class.save_plotly_figure(fig,
                                          "consolidated",
                                          height=TALL_FIG_HEIGHT*2,
                                          width=4960,
                                          scale=C.SCALE,
                                          save_final=True,
                                          save_eps=False,
                                          save_png=False)

    def speed_and_time_to_start_cross_country(self, df_mapping, font_size_captions=40, x_axis_title_height=150,
                                              legend_x=0.81, legend_y=0.98, legend_spacing=0.02):
        logger.info("Plotting speed_and_time_to_start_cross")
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[27]
        avg_time = data_tuple[28]
        no_of_crossing = data_tuple[35]

        # Check if both 'speed' and 'time' are valid dictionaries
        if avg_speed is None or avg_time is None:
            raise ValueError("Either 'speed' or 'time' returned None, please check the input data or calculations.")

        # Remove the ones where there is data missing for a specific country and condition
        common_keys = avg_speed.keys() & avg_time.keys()

        # Retain only the key-value pairs where the key is present in both dictionaries
        avg_speed = {key: avg_speed[key] for key in common_keys}
        avg_time = {key: avg_time[key] for key in common_keys}

        # Now populate the final_dict with country-wise data
        for country_condition, speed in tqdm(avg_speed.items()):
            if no_of_crossing[country_condition] < common.get_configs("min_crossing_detect"):
                continue
            country, condition = country_condition.split('_')

            # Get the iso3 from the mapping file
            iso_code = metadata_class.get_value(df=df_mapping,
                                                column_name1="country",
                                                column_value1=country,
                                                column_name2=None,
                                                column_value2=None,
                                                target_column="iso3")

            if country and iso_code is not None:
                # Initialise the country's dictionary if not already present
                if f'{country}' not in final_dict:
                    final_dict[f"{country}"] = {
                        "speed_0": None, "speed_1": None, "time_0": None, "time_1": None,
                        "country": country, "iso3": iso_code}

                # Populate the corresponding speed and time based on the condition
                final_dict[f"{country}"][f"speed_{condition}"] = speed
                if f'{country}_{condition}' in avg_time:
                    final_dict[f"{country}"][f"time_{condition}"] = avg_time[f'{country}_{condition}']

        # Extract all valid speed_0 and speed_1 values along with their corresponding countries
        diff_speed_values = [(f'{country}', abs(data['speed_0'] - data['speed_1']))
                             for country, data in final_dict.items()
                             if data['speed_0'] is not None and data['speed_1'] is not None]

        if diff_speed_values:
            # Sort the list by the absolute difference and get the top 5 and bottom 5
            sorted_diff_speed_values = sorted(diff_speed_values, key=lambda x: x[1], reverse=True)

            top_5_max_speed = sorted_diff_speed_values[:5]  # Top 5 maximum differences
            top_5_min_speed = sorted_diff_speed_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("Top 5 country with max |speed_0 - speed_1| differences:")
            for country, diff in top_5_max_speed:
                logger.info(f"{grouping_class.format_city_state(country)}: {diff}")

            logger.info("Top 5 cities with min |speed_0 - speed_1| differences:")
            for country, diff in top_5_min_speed:
                logger.info(f"{grouping_class.format_city_state(country)}: {diff}")
        else:
            logger.info("No valid speed_0 and speed_1 values found for comparison.")

        # Extract all valid time_0 and time_1 values along with their corresponding countries
        diff_time_values = [(country, abs(data['time_0'] - data['time_1']))
                            for country, data in final_dict.items()
                            if data['time_0'] is not None and data['time_1'] is not None]

        if diff_time_values:
            sorted_diff_time_values = sorted(diff_time_values, key=lambda x: x[1], reverse=True)

            top_5_max = sorted_diff_time_values[:5]  # Top 5 maximum differences
            top_5_min = sorted_diff_time_values[-5:]  # Top 5 minimum differences (including possible zeroes)

            logger.info("Top 5 cities with max |time_0 - time_1| differences:")
            for country, diff in top_5_max:
                logger.info(f"{grouping_class.format_city_state(country)}: {diff}")

            logger.info("Top 5 cities with min |time_0 - time_1| differences:")
            for country, diff in top_5_min:
                logger.info(f"{grouping_class.format_city_state(country)}: {diff}")
        else:
            logger.info("No valid time_0 and time_1 values found for comparison.")

        # Filtering out entries where entries is None
        filtered_dict_s_0 = {country: info for country, info in final_dict.items() if info["speed_0"] is not None}
        filtered_dict_s_1 = {country: info for country, info in final_dict.items() if info["speed_1"] is not None}
        filtered_dict_t_0 = {country: info for country, info in final_dict.items() if info["time_0"] is not None}
        filtered_dict_t_1 = {country: info for country, info in final_dict.items() if info["time_1"] is not None}

        # Find country with max and min speed_0 and speed_1
        if filtered_dict_s_0:
            max_speed_country_0 = max(filtered_dict_s_0, key=lambda country: filtered_dict_s_0[country]["speed_0"])
            min_speed_country_0 = min(filtered_dict_s_0, key=lambda country: filtered_dict_s_0[country]["speed_0"])
            max_speed_value_0 = filtered_dict_s_0[max_speed_country_0]["speed_0"]
            min_speed_value_0 = filtered_dict_s_0[min_speed_country_0]["speed_0"]

            logger.info(f"Country with max speed at day: {grouping_class.format_city_state(max_speed_country_0)} with speed of {max_speed_value_0} m/s")  # noqa:E501
            logger.info(f"Country with min speed at day: {grouping_class.format_city_state(min_speed_country_0)} with speed of {min_speed_value_0} m/s")  # noqa:E501

        if filtered_dict_s_1:
            max_speed_country_1 = max(filtered_dict_s_1, key=lambda country: filtered_dict_s_1[country]["speed_1"])
            min_speed_country_1 = min(filtered_dict_s_1, key=lambda country: filtered_dict_s_1[country]["speed_1"])
            max_speed_value_1 = filtered_dict_s_1[max_speed_country_1]["speed_1"]
            min_speed_value_1 = filtered_dict_s_1[min_speed_country_1]["speed_1"]

            logger.info(f"Country with max speed at night: {grouping_class.format_city_state(max_speed_country_1)} with speed of {max_speed_value_1} m/s")  # noqa:E501
            logger.info(f"Country with min speed at night: {grouping_class.format_city_state(min_speed_country_1)} with speed of {min_speed_value_1} m/s")  # noqa:E501

        # Find country with max and min time_0 and time_1
        if filtered_dict_t_0:
            max_time_country_0 = max(filtered_dict_t_0, key=lambda country: filtered_dict_t_0[country]["time_0"])
            min_time_country_0 = min(filtered_dict_t_0, key=lambda country: filtered_dict_t_0[country]["time_0"])
            max_time_value_0 = filtered_dict_t_0[max_time_country_0]["time_0"]
            min_time_value_0 = filtered_dict_t_0[min_time_country_0]["time_0"]

            logger.info(f"Country with max time at day: {grouping_class.format_city_state(max_time_country_0)} with time of {max_time_value_0} s")  # noqa:E501
            logger.info(f"Country with min time at day: {grouping_class.format_city_state(min_time_country_0)} with time of {min_time_value_0} s")  # noqa:E501

        if filtered_dict_t_1:
            max_time_country_1 = max(filtered_dict_t_1, key=lambda country: filtered_dict_t_1[country]["time_1"])
            min_time_country_1 = min(filtered_dict_t_1, key=lambda country: filtered_dict_t_1[country]["time_1"])
            max_time_value_1 = filtered_dict_t_1[max_time_country_1]["time_1"]
            min_time_value_1 = filtered_dict_t_1[min_time_country_1]["time_1"]

            logger.info(f"Country with max time at night: {grouping_class.format_city_state(max_time_country_1)} with time of {max_time_value_1} s")  # noqa:E501
            logger.info(f"Country with min time at night: {grouping_class.format_city_state(min_time_country_1)} with time of {min_time_value_1} s")  # noqa:E501

        # Extract valid speed and time values and calculate statistics
        speed_0_values = [data['speed_0'] for data in final_dict.values() if pd.notna(data['speed_0'])]
        speed_1_values = [data['speed_1'] for data in final_dict.values() if pd.notna(data['speed_1'])]
        time_0_values = [data['time_0'] for data in final_dict.values() if pd.notna(data['time_0'])]
        time_1_values = [data['time_1'] for data in final_dict.values() if pd.notna(data['time_1'])]

        if speed_0_values:
            mean_speed_0 = statistics.mean(speed_0_values)
            sd_speed_0 = statistics.stdev(speed_0_values) if len(speed_0_values) > 1 else 0
            logger.info(f"Mean of speed during day time: {mean_speed_0}")
            logger.info(f"Standard deviation of speed during day time: {sd_speed_0}")
        else:
            logger.error("No valid speed during day time values found.")

        if speed_1_values:
            mean_speed_1 = statistics.mean(speed_1_values)
            sd_speed_1 = statistics.stdev(speed_1_values) if len(speed_1_values) > 1 else 0
            logger.info(f"Mean of speed during night time: {mean_speed_1}")
            logger.info(f"Standard deviation of speed during night time: {sd_speed_1}")
        else:
            logger.error("No valid speed during night time values found.")

        if time_0_values:
            mean_time_0 = statistics.mean(time_0_values)
            sd_time_0 = statistics.stdev(time_0_values) if len(time_0_values) > 1 else 0
            logger.info(f"Mean of time during day time: {mean_time_0}")
            logger.info(f"Standard deviation of time during day time: {sd_time_0}")
        else:
            logger.error("No valid time during day time values found.")

        if time_1_values:
            mean_time_1 = statistics.mean(time_1_values)
            sd_time_1 = statistics.stdev(time_1_values) if len(time_1_values) > 1 else 0
            logger.info(f"Mean of time during night time: {mean_time_1}")
            logger.info(f"Standard deviation of time during night time: {sd_time_1}")
        else:
            logger.error("No valid time during night time values found.")

        # Extract country, condition, and count_ from the info dictionary
        countries, conditions_, counts = [], [], []
        for key, value in tqdm(avg_time.items()):
            country, condition = key.split('_')
            countries.append(f'{country}')
            conditions_.append(condition)
            counts.append(value)

        # Sort the list of tuples by country name
        countries_ordered = sorted(final_dict, key=lambda x: x[0])

        # Extract the desired values from the sorted list
        day_avg_speed = [final_dict[country]['speed_0'] for country in countries_ordered]
        night_avg_speed = [final_dict[country]['speed_1'] for country in countries_ordered]
        day_time_dict = [final_dict[country]['time_0'] for country in countries_ordered]
        night_time_dict = [final_dict[country]['time_1'] for country in countries_ordered]

        # Ensure that plotting uses cities_ordered
        assert len(countries_ordered) == len(day_avg_speed) == len(night_avg_speed) == len(
            day_time_dict) == len(night_time_dict), "Lengths of lists don't match!"

        # Determine how many cities will be in each column
        num_cities_per_col = len(countries_ordered) // 2 + len(countries_ordered) % 2  # Split cities into two groups
        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * C.BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col * 2, cols=2,  # Two columns
            vertical_spacing=0,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[2.0] * (num_cities_per_col * 2),
        )

        # Plot left column (first half of cities)
        for i, country in enumerate(countries_ordered[:num_cities_per_col]):
            iso_code = metadata_class.get_value(df_mapping, "country", country, None, None, "iso3")
            # build up textual label for left column
            iso2 = iso_class.iso3_to_iso2(iso_code)
            # country = Analysis.iso2_to_flag(iso2) + " " + iso_code + " " + country
            country = iso_class.iso2_to_flag(iso2) + " " + country
            # Row for speed (Day and Night)
            row = 2 * i + 1
            if day_avg_speed[i] is not None and night_avg_speed[i] is not None:
                value = (day_avg_speed[i] + night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during day", marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during night",
                    marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_avg_speed[i] is not None:  # Only day data available
                value = (day_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during day", marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_avg_speed[i] is not None:  # Only night data available
                value = (night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during night",
                    marker=dict(color=C.BAR_COLOR_2), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            # Row for time (Day and Night)
            row = 2 * i + 2
            if day_time_dict[i] is not None and night_time_dict[i] is not None:
                value = (day_time_dict[i] + night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during day", marker=dict(color=C.BAR_COLOR_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during night", marker=dict(color=C.BAR_COLOR_4), text=[''],
                    textposition='auto', showlegend=False), row=row, col=1)

            elif day_time_dict[i] is not None:  # Only day time data available
                value = (day_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during day", marker=dict(color=C.BAR_COLOR_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_time_dict[i] is not None:  # Only night time data available
                value = (night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during night", marker=dict(color=C.BAR_COLOR_4),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

        # Similarly for the right column
        for i, country in enumerate(countries_ordered[num_cities_per_col:]):
            iso_code = metadata_class.get_value(df_mapping, "country", country, None, None, "iso3")
            row = 2 * i + 1
            idx = num_cities_per_col + i
            # build up textual label for left column
            iso2 = iso_class.iso3_to_iso2(iso_code)
            # country = Analysis.iso2_to_flag(iso2) + " " + iso_code + " " + country
            country = iso_class.iso2_to_flag(iso2) + " " + country
            if day_avg_speed[idx] is not None and night_avg_speed[idx] is not None:
                value = (day_avg_speed[idx] + night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during day", marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during night", marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_avg_speed[idx] is not None:
                value = (day_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during day", marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_avg_speed[idx] is not None:
                value = (night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during night", marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            row = 2 * i + 2
            if day_time_dict[idx] is not None and night_time_dict[idx] is not None:
                value = (day_time_dict[idx] + night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during day", marker=dict(color=C.BAR_COLOR_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during night", marker=dict(color=C.BAR_COLOR_4), text=[''],
                    textposition='inside', showlegend=False), row=row, col=2)

            elif day_time_dict[idx] is not None:  # Only day time data available
                value = (day_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during day", marker=dict(color=C.BAR_COLOR_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_time_dict[idx] is not None:  # Only night time data available
                value = (night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during night", marker=dict(color=C.BAR_COLOR_4),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_time = max([
            (day_time_dict[i] if day_time_dict[i] is not None else 0) +
            (night_time_dict[i] if night_time_dict[i] is not None else 0)
            for i in range(len(countries_ordered))
        ]) if countries_ordered else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(countries) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=False
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_time], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=False
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(
            title=dict(text="Mean speed of crossing (in m/s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=1
        )
        fig.update_xaxes(
            title=dict(text="Mean speed of crossing (in m/s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=2
        )
        fig.update_xaxes(
            title=dict(text="Mean time to start crossing (in s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=num_cities_per_col * 2,
            col=1,
        )
        fig.update_xaxes(
            title=dict(text="Mean time to start crossing (in s)", font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=num_cities_per_col * 2,
            col=2
        )

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=TALL_FIG_HEIGHT*2, width=4960, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Set the x-axis range to cover the values you want in x_grid_values
        # TODO: move away from hardcoded xtick values
        x_grid_values = [2, 4, 6, 8, 10, 12, 14, 16, 18]

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x', yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x2', yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Define the legend items
        legend_items = [
            {"name": "Mean speed of crossing during day (in m/s)", "color": C.BAR_COLOR_1},
            {"name": "Mean speed of crossing during night (in m/s)", "color": C.BAR_COLOR_2},
            {"name": "Mean time to start crossing during day (in s)", "color": C.BAR_COLOR_3},
            {"name": "Mean time to start crossing during night (in s) ", "color": C.BAR_COLOR_4},
        ]

        # Add the vertical legends at the top and bottom
        layout_class.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
                                                     spacing=legend_spacing, font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=1, x1=0.495, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.505, y0=1, x1=1, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        fig.update_yaxes(
            tickfont=dict(size=14, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=10, r=10, t=x_axis_title_height, b=x_axis_title_height))
        plots_io_class.save_plotly_figure(fig, "consolidated", height=TALL_FIG_HEIGHT*2, width=4960, scale=C.SCALE,
                                          save_final=True, save_eps=False)

    def plot_crossing_without_traffic_light(self, df_mapping, font_size_captions=40, x_axis_title_height=150,
                                            legend_x=0.92, legend_y=0.015, legend_spacing=0.02):
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        without_trf_light = data_tuple[28]
        # Now populate the final_dict with city-wise speed data
        for city_condition, count in without_trf_light.items():
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:
                # Initialise the city's dictionary if not already present
                if f"{city}_{lat}_{long}" not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {"without_trf_light_0": None, "without_trf_light_1": None,
                                                          "country": country, "iso": iso_code}

                # normalise by total time and total number of detected persons
                total_time = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "total_time")
                person = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "person")
                count = count / total_time / person

                # Populate the corresponding speed based on the condition
                final_dict[f"{city}_{lat}_{long}"][f"without_trf_light_{condition}"] = count

        # Multiply each of the numeric speed values by 10^6
        for city_key, data in final_dict.items():
            for key, value in data.items():
                # Only modify keys that represent speed values
                if key.startswith("without_trf_light") and value is not None:
                    data[key] = round(value * 10**6, 2)

        cities_ordered = sorted(
            final_dict.keys(),
            key=lambda city: dataset_stats.safe_average([
                final_dict[city]["without_trf_light_0"],
                final_dict[city]["without_trf_light_1"]
            ]),
            reverse=True
        )

        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        # Prepare data for day and night stacking
        day_crossing = [final_dict[city]['without_trf_light_0'] for city in cities_ordered]
        night_crossing = [final_dict[city]['without_trf_light_1'] for city in cities_ordered]

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups
        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * C.BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city_new, lat, long = city.split('_')
            iso_code = metadata_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = grouping_class.process_city_string(city, df_mapping)

            city = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            if day_crossing[i] is not None and night_crossing[i] is not None:
                value = round((day_crossing[i] + night_crossing[i])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=1)

            elif day_crossing[i] is not None:  # Only day data available
                value = (day_crossing[i])
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_crossing[i] is not None:  # Only night data available
                value = (night_crossing[i])
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=C.BAR_COLOR_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, lat, long = city.split('_')
            iso_code = metadata_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = grouping_class.process_city_string(city, df_mapping)

            city = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            idx = num_cities_per_col + i
            if day_crossing[idx] is not None and night_crossing[idx] is not None:
                value = round((day_crossing[idx] + night_crossing[idx])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_crossing[idx] is not None:
                value = (day_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_crossing[idx] is not None:
                value = (night_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=C.BAR_COLOR_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_crossing[i] if day_crossing[i] is not None else 0) +
            (night_crossing[i] if night_crossing[i] is not None else 0)
            for i in range(len(cities))
        ]) if cities else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=True
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=True
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(title=dict(text="Road crossings without traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=1)

        fig.update_xaxes(title=dict(text="Road crossings without traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=2)

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=TALL_FIG_HEIGHT, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Manually add gridlines using `shapes`
        x_grid_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600]  # Define the gridline positions on the x-axis

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x', yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x2', yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Define the legend items
        legend_items = [
            {"name": "Day", "color": C.BAR_COLOR_1},
            {"name": "Night", "color": C.BAR_COLOR_2},
        ]

        # Add the vertical legends at the top and bottom
        layout_class.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
                                                     spacing=legend_spacing, font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=1, x1=0.495, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.505, y0=1, x1=1, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Split cities into left and right columns
        left_column_cities = cities_ordered[:num_cities_per_col]
        right_column_cities = cities_ordered[num_cities_per_col:]

        # Initialise variables for dynamic y positioning for both columns
        current_row_left = 1  # Start from the first row for the left column
        current_row_right = 1  # Start from the first row for the right column
        y_position_map_left = {}  # Store y positions for each country (left column)
        y_position_map_right = {}  # Store y positions for each country (right column)

        # Calculate the y positions dynamically for the left column
        for city in left_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_left:  # Add the country label once per country
                y_position_map_left[country] = 1 - (current_row_left - 1) / (len(left_column_cities) * 2)

            current_row_left += 2  # Increment the row for each city (speed and time take two rows)

        # Calculate the y positions dynamically for the right column
        for city in right_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_right:  # Add the country label once per country
                y_position_map_right[country] = 1 - (current_row_right - 1) / (len(right_column_cities) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        fig.update_yaxes(
            tickfont=dict(size=12, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=x_axis_title_height, b=180))
        plots_io_class.save_plotly_figure(fig,
                                          "crossings_without_traffic_equipment_avg",
                                          width=2480,
                                          height=TALL_FIG_HEIGHT,
                                          scale=C.SCALE,
                                          save_eps=False,
                                          save_final=True)

    def plot_crossing_with_traffic_light(self, df_mapping, font_size_captions=40, x_axis_title_height=150,
                                         legend_x=0.92, legend_y=0.015, legend_spacing=0.02):
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        with_trf_light = data_tuple[27]
        # Now populate the final_dict with city-wise speed data
        for city_condition, count in with_trf_light.items():
            city, lat, long, condition = city_condition.split('_')

            # Get the country from the previously stored city_country_map
            country = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:
                # Initialise the city's dictionary if not already present
                if f"{city}_{lat}_{long}" not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {"with_trf_light_0": None, "with_trf_light_1": None,
                                                          "country": country, "iso": iso_code}

                # normalise by total time and total number of detected persons
                total_time = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "total_time")
                person = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "person")
                count = count / total_time / person

                # Populate the corresponding speed based on the condition
                final_dict[f"{city}_{lat}_{long}"][f"with_trf_light_{condition}"] = count

        # Multiply each of the numeric speed values by 10^6
        for city_key, data in final_dict.items():
            for key, value in data.items():
                # Only modify keys that represent speed values
                if key.startswith("with_trf_light") and value is not None:
                    data[key] = round(value * 10**6, 2)

        cities_ordered = sorted(
            final_dict.keys(),
            key=lambda city: dataset_stats.safe_average([
                final_dict[city]["with_trf_light_0"],
                final_dict[city]["with_trf_light_1"]
            ]),
            reverse=True
        )

        # Extract unique cities
        cities = list(set([key.split('_')[0] for key in final_dict.keys()]))

        # Prepare data for day and night stacking
        day_crossing = [final_dict[city]['with_trf_light_0'] for city in cities_ordered]
        night_crossing = [final_dict[city]['with_trf_light_1'] for city in cities_ordered]

        # # Ensure that plotting uses cities_ordered
        # assert len(cities_ordered) == len(day_crossing) == len(night_crossing), "Lengths of lists don't match!"

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups
        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * C.BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city_new, lat, long = city.split('_')
            iso_code = metadata_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = grouping_class.process_city_string(city, df_mapping)

            city = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            if day_crossing[i] is not None and night_crossing[i] is not None:
                value = round((day_crossing[i] + night_crossing[i])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_crossing[i] is not None:  # Only day data available
                value = (day_crossing[i])
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_crossing[i] is not None:  # Only night data available
                value = (night_crossing[i])
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=C.BAR_COLOR_2),
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, lat, long = city.split('_')
            iso_code = metadata_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = grouping_class.process_city_string(city, df_mapping)

            city = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            idx = num_cities_per_col + i
            if day_crossing[idx] is not None and night_crossing[idx] is not None:
                value = round((day_crossing[idx] + night_crossing[idx])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=C.BAR_COLOR_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_crossing[idx] is not None:
                value = (day_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=C.BAR_COLOR_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_crossing[idx] is not None:
                value = (night_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=C.BAR_COLOR_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value_speed = max([
            (day_crossing[i] if day_crossing[i] is not None else 0) +
            (night_crossing[i] if night_crossing[i] is not None else 0)
            for i in range(len(cities))
        ]) if cities else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=True
                )

            # Update x-axis for the right column (top for speed, bottom for time)
            if i % 2 == 1:  # Odd rows (representing speed)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=True
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value_speed], row=i, col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=True
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(title=dict(text="Road crossings with traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=1)

        fig.update_xaxes(title=dict(text="Road crossings with traffic signals (normalised)",
                         font=dict(size=font_size_captions)), tickfont=dict(size=font_size_captions), ticks='outside',
                         ticklen=10, tickwidth=2, tickcolor='black', row=1, col=2)

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', barmode='stack',
            height=TALL_FIG_HEIGHT, width=2480, showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150), bargap=0, bargroupgap=0
        )

        # Manually add gridlines using `shapes`
        x_grid_values = [50, 100, 150, 200, 250]  # Define the gridline positions on the x-axis

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x', yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=1,  # Set the position of the gridlines
                xref='x2', yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Define the legend items
        legend_items = [
            {"name": "Day", "color": C.BAR_COLOR_1},
            {"name": "Night", "color": C.BAR_COLOR_2},
        ]

        # Add the vertical legends at the top and bottom
        layout_class.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
                                                     spacing=legend_spacing, font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=1, x1=0.495, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.505, y0=1, x1=1, y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        # Split cities into left and right columns
        left_column_cities = cities_ordered[:num_cities_per_col]
        right_column_cities = cities_ordered[num_cities_per_col:]

        # Initialise variables for dynamic y positioning for both columns
        current_row_left = 1  # Start from the first row for the left column
        current_row_right = 1  # Start from the first row for the right column
        y_position_map_left = {}  # Store y positions for each country (left column)
        y_position_map_right = {}  # Store y positions for each country (right column)

        # Calculate the y positions dynamically for the left column
        for city in left_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_left:  # Add the country label once per country
                y_position_map_left[country] = 1 - (current_row_left - 1) / (len(left_column_cities) * 2)

            current_row_left += 2  # Increment the row for each city (speed and time take two rows)

        # Calculate the y positions dynamically for the right column
        for city in right_column_cities:
            country = final_dict[city]['iso']

            if country not in y_position_map_right:  # Add the country label once per country
                y_position_map_right[country] = 1 - (current_row_right - 1) / (len(right_column_cities) * 2)

            current_row_right += 2  # Increment the row for each city (speed and time take two rows)

        fig.update_yaxes(
            tickfont=dict(size=12, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=100, t=x_axis_title_height, b=180))
        plots_io_class.save_plotly_figure(fig,
                                          "crossings_with_traffic_equipment_avg",
                                          width=2480,
                                          height=TALL_FIG_HEIGHT,
                                          scale=C.SCALE,
                                          save_eps=False,
                                          save_final=True)

