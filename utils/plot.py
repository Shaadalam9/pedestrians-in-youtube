import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
import os
from .values import Values
from .wrappers import Wrappers
import shutil
import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from scipy.spatial import KDTree
import statistics
import itertools


# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

values_class = Values()
wrapper_class = Wrappers()

# File to store the city coordinates
file_results = 'results.pickle'

# Colours in graphs
bar_colour_1 = 'rgb(251, 180, 174)'
bar_colour_2 = 'rgb(179, 205, 227)'
bar_colour_3 = 'rgb(204, 235, 197)'
bar_colour_4 = 'rgb(222, 203, 228)'

# Consts
BASE_HEIGHT_PER_ROW = 30  # Adjust as needed
FLAG_SIZE = 12
TEXT_SIZE = 12
SCALE = 1  # scale=3 hangs often


class Plots():
    def __init__(self) -> None:
        pass

    def add_vertical_legend_annotations(self, fig, legend_items, x_position, y_start, spacing=0.03, font_size=50):
        for i, item in enumerate(legend_items):
            fig.add_annotation(
                x=x_position,  # Use the x_position provided by the user
                y=y_start - i * spacing,  # Adjust vertical position based on index and spacing
                xref='paper', yref='paper', showarrow=False,
                text=f'<span style="color:{item["color"]};">&#9632;</span> {item["name"]}',  # noqa:E501
                font=dict(size=font_size),
                xanchor='left', align='left'  # Ensure the text is left-aligned
            )

    def save_plotly_figure(self, fig, filename, width=1600, height=900, scale=SCALE, save_final=True, save_png=True,
                           save_eps=True):
        """
        Saves a Plotly figure as HTML, PNG, SVG, and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): Width of the PNG and EPS images in pixels. Defaults to 1600.
            height (int, optional): Height of the PNG and EPS images in pixels. Defaults to 900.
            scale (int, optional): Scaling factor for the PNG image. Defaults to 3.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # Create directory if it doesn't exist
        output_final = os.path.join(common.root_dir, 'figures')
        os.makedirs(common.output_dir, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)

        # Save as HTML
        logger.info(f"Saving html file for {filename}.")
        py.offline.plot(fig, filename=os.path.join(common.output_dir, filename + ".html"))
        # also save the final figure
        if save_final:
            py.offline.plot(fig, filename=os.path.join(output_final, filename + ".html"),  auto_open=False)

        try:
            # Save as PNG
            if save_png:
                logger.info(f"Saving png file for {filename}.")
                fig.write_image(os.path.join(common.output_dir, filename + ".png"), width=width, height=height,
                                scale=scale)
                # also save the final figure
                if save_final:
                    shutil.copy(os.path.join(common.output_dir, filename + ".png"),
                                os.path.join(output_final, filename + ".png"))

            # Save as EPS
            if save_eps:
                logger.info(f"Saving eps file for {filename}.")
                fig.write_image(os.path.join(common.output_dir, filename + ".eps"), width=width, height=height)
                # also save the final figure
                if save_final:
                    shutil.copy(os.path.join(common.output_dir, filename + ".eps"),
                                os.path.join(output_final, filename + ".eps"))
        except ValueError:
            logger.error(f"Value error raised when attempted to save image {filename}.")

    def stack_plot(self, df_mapping, order_by, metric, data_view, title_text, filename, analysis_level="city",
                   font_size_captions=40, x_axis_title_height=110, legend_x=0.92, legend_y=0.015, legend_spacing=0.02,
                   left_margin=10, right_margin=10):
        """
        Plots a stacked bar graph based on the provided data and configuration.

        Parameters:
            df_mapping (dict): A dictionary mapping categories to their respective DataFrames.
            order_by (str): Criterion to order the bars, e.g., 'alphabetical' or 'average'.
            metric (str): The metric to visualize, such as 'speed' or 'time'.
            data_view (str): Determines which subset of data to visualise, such as 'day', 'night', or 'combined'.
            title_text (str): The title of the plot.
            filename (str): The name of the file to save the plot as.
            font_size_captions (int, optional): Font size for captions. Default is 40.
            x_axis_title_height (int, optional): Vertical space for x-axis title. Default is 110.
            legend_x (float, optional): X position of the legend. Default is 0.92.
            legend_y (float, optional): Y position of the legend. Default is 0.015.
            legend_spacing (float, optional): Spacing between legend entries. Default is 0.02.

        Returns:
            None
        """

        # Define log messages in a structured way
        log_messages = {
            ("alphabetical", "speed", "day"): "Plotting speed to cross by alphabetical order during day time.",
            ("alphabetical", "speed", "night"): "Plotting speed to cross by alphabetical order during night time.",
            ("alphabetical", "speed", "combined"): "Plotting speed to cross by alphabetical order.",
            ("alphabetical", "time", "day"): "Plotting time to start cross by alphabetical order during day time.",
            ("alphabetical", "time", "night"): "Plotting time to start cross by alphabetical order during night time.",
            ("alphabetical", "time", "combined"): "Plotting time to start cross by alphabetical order.",
            ("average", "speed", "day"): "Plotting speed to cross by average during day time.",
            ("average", "speed", "night"): "Plotting speed to cross by averageduring night time.",
            ("average", "speed", "combined"): "Plotting speed to cross by average.",
            ("average", "time", "day"): "Plotting time to start cross by average during day time.",
            ("average", "time", "night"): "Plotting time to start cross by average during night time.",
            ("average", "time", "combined"): "Plotting time to start cross by average."
        }

        message = log_messages.get((order_by, metric, data_view))
        final_dict = {}

        if message:
            logger.info(message)

        # Map metric names to their index in the data tuple
        if analysis_level == "city":
            metric_index_map = {
                "speed": 25,
                "time": 24
            }
        elif analysis_level == "country":
            metric_index_map = {
                "speed": 27,
                "time": 28
                }

        if metric not in metric_index_map:
            raise ValueError(f"Unsupported metric: {metric}")

        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        metric_data = data_tuple[metric_index_map[metric]]

        if metric_data is None:
            raise ValueError(f"'{metric}' returned None, please check the input data or calculations.")

        # Clean NaNs
        metric_data = {
            key: value for key, value in metric_data.items()
            if not (isinstance(value, float) and math.isnan(value))
        }

        if analysis_level == "city":
            # Now populate the final_dict with city-wise speed data
            for city_condition, _ in tqdm(metric_data.items()):
                city, lat, long, condition = city_condition.split('_')

                # Get the country from the previously stored city_country_map
                country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
                iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")

                if country or iso_code is not None:
                    # Initialise the city's dictionary if not already present
                    if f'{city}_{lat}_{long}' not in final_dict:
                        final_dict[f'{city}_{lat}_{long}'] = {f"{metric}_0": None,
                                                              f"{metric}_1": None,
                                                              "country": country,
                                                              "iso": iso_code}

                    # Populate the corresponding speed based on the condition
                    final_dict[f'{city}_{lat}_{long}'][f"{metric}_{condition}"] = _

        if analysis_level == "country":
            for country_condition, _ in tqdm(metric_data.items()):
                country, condition = country_condition.split('_')

                # Get the iso3 from the mapping file
                iso_code = values_class.get_value(df=df_mapping,
                                                  column_name1="country",
                                                  column_value1=country,
                                                  column_name2=None,
                                                  column_value2=None,
                                                  target_column="iso3")

                if country is not None or iso_code is not None:
                    # Initialise the city's dictionary if not already present
                    if f'{country}' not in final_dict:
                        final_dict[f'{country}'] = {f"{metric}_0": None, f"{metric}_1": None,
                                                    "country": country, "iso3": iso_code}
                    # Populate the corresponding speed based on the condition
                    final_dict[f'{country}'][f"{metric}_{condition}"] = _  # type: ignore

        if order_by == "alphabetical":
            if data_view == "day":
                cities_ordered = sorted(
                    [
                        city for city in final_dict.keys()
                        if (final_dict[city].get(f"{metric}_0") or 0) >= 0.005
                    ],
                    key=lambda city: (final_dict[city].get("iso") or "")
                )
            elif data_view == "night":
                cities_ordered = sorted(
                    [
                        city for city in final_dict.keys()
                        if (final_dict[city].get(f"{metric}_1") or 0) >= 0.005
                    ],
                    key=lambda city: (final_dict[city].get("iso") or "")
                )
            else:
                cities_ordered = sorted(
                    [
                        city for city in final_dict.keys()
                        if (((final_dict[city].get(f"{metric}_0") or 0) + (final_dict[city].get(f"{metric}_1") or 0)) / 2) >= 0.005  # noqa:E501
                    ],
                    key=lambda city: (final_dict[city].get("iso") or "")
                )

        elif order_by == "average":
            if data_view == "day":
                cities_ordered = sorted(
                    [
                        city for city in final_dict.keys()
                        if (final_dict[city].get(f"{metric}_0") or 0) >= 0.005
                    ],
                    key=lambda city: final_dict[city].get(f"{metric}_0") or 0,
                    reverse=True
                )

            elif data_view == "night":
                cities_ordered = sorted(
                    [
                        city for city in final_dict.keys()
                        if (final_dict[city].get(f"{metric}_1") or 0) >= 0.005
                    ],
                    key=lambda city: final_dict[city].get(f"{metric}_1") or 0,
                    reverse=True
                )

            else:
                cities_ordered = sorted(
                    [
                        city for city in final_dict.keys()
                        if (((final_dict[city].get(f"{metric}_0") or 0) + (final_dict[city].get(f"{metric}_1") or 0)) / 2) >= 0.005  # noqa:E501  # type: ignore
                    ],
                    key=lambda city: (
                        ((final_dict[city].get(f"{metric}_0") or 0) + (final_dict[city].get(f"{metric}_1") or 0)) / 2  # noqa:E501  # type: ignore
                    ), reverse=True
                )

        if len(cities_ordered) == 0:
            return

        # Prepare data for day and night stacking
        day_key = f"{metric}_0"
        night_key = f"{metric}_1"

        if data_view == "combined":
            day_values = [final_dict[country][day_key] for country in cities_ordered]
            night_values = [final_dict[country][night_key] for country in cities_ordered]
        elif data_view == "day":
            day_values = [final_dict[country][day_key] for country in cities_ordered]
            night_values = [0 for country in cities_ordered]
        elif data_view == "night":
            day_values = [0 for country in cities_ordered]
            night_values = [final_dict[country][night_key] for country in cities_ordered]

        # Determine how many cities will be in each column
        num_cities_per_col = len(cities_ordered) // 2 + len(cities_ordered) % 2  # Split cities into two groups

        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_cities_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col, cols=2,  # Two columns
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city_new, lat, long = city.split('_')
            city = wrapper_class.process_city_string(city, df_mapping)

            if order_by == "average":
                iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
                city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city  # type: ignore

            # Row for speed (Day and Night)
            row = i + 1
            if day_values[i] is not None and night_values[i] is not None:
                if data_view == "combined":
                    value = (day_values[i] + night_values[i])/2
                else:
                    value = (day_values[i] + night_values[i])

                fig.add_trace(go.Bar(
                    x=[day_values[i]],
                    y=[f'{city} {value:.2f}'],
                    orientation='h',
                    name=f"{city} {metric} during day",
                    marker=dict(color=bar_colour_1),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=14,
                                  color='white')),
                                  row=row,
                                  col=1)

                fig.add_trace(go.Bar(
                    x=[night_values[i]],
                    y=[f'{city} {value:.2f}'],
                    orientation='h',
                    name=f"{city} {metric} during night",
                    marker=dict(color=bar_colour_2),
                    text=[''],
                    textposition='inside',
                    showlegend=False),
                    row=row,
                    col=1)

            elif day_values[i] is not None:  # Only day data available
                value = day_values[i]
                fig.add_trace(go.Bar(
                    x=[day_values[i]],
                    y=[f'{city} {value:.2f}'],
                    orientation='h',
                    name=f"{city} {metric} during day",
                    marker=dict(color=bar_colour_1),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=14,
                                  color='white')),
                                  row=row,
                                  col=1)

            elif night_values[i] is not None:  # Only night data available
                value = night_values[i]
                fig.add_trace(go.Bar(
                    x=[night_values[i]],
                    y=[f'{city} {value:.2f}'],
                    orientation='h',
                    name=f"{city} {metric} during night",
                    marker=dict(color=bar_colour_2),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=14,
                                  color='white')),
                                  row=row,
                                  col=1)

        # Plot right column (second half of cities)
        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, lat, long = city.split('_')
            city = wrapper_class.process_city_string(city, df_mapping)

            if order_by == "average":
                iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
                city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city  # type: ignore

            # Row for speed (Day and Night)
            row = i + 1
            idx = num_cities_per_col + i
            if day_values[idx] is not None and night_values[idx] is not None:
                if data_view == "combined":
                    value = (day_values[i] + night_values[i])/2
                else:
                    value = (day_values[i] + night_values[i])

                fig.add_trace(go.Bar(
                    x=[day_values[idx]],
                    y=[f'{city} {value:.2f}'],
                    orientation='h',
                    name=f"{city} {metric} during day",
                    marker=dict(color=bar_colour_1),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=14,
                                  color='white')),
                                  row=row,
                                  col=2)

                fig.add_trace(go.Bar(
                    x=[night_values[idx]],
                    y=[f'{city} {value:.2f}'],
                    orientation='h',
                    name=f"{city} {metric} during night",
                    marker=dict(color=bar_colour_2),
                    text=[''],
                    textposition='inside',
                    showlegend=False),
                    row=row,
                    col=2)

            elif day_values[idx] is not None:
                value = day_values[idx]
                fig.add_trace(go.Bar(
                    x=[day_values[idx]],
                    y=[f'{city} {value:.2f}'],
                    orientation='h',
                    name=f"{city} {metric} during day",
                    marker=dict(color=bar_colour_1),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=14,
                                  color='white')),
                                  row=row,
                                  col=2)

            elif night_values[idx] is not None:
                value = night_values[idx]
                fig.add_trace(go.Bar(
                    x=[night_values[idx]],
                    y=[f'{city} {value:.2f}'],
                    orientation='h',
                    name=f"{city} {metric} during night",
                    marker=dict(color=bar_colour_2),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=14,
                                  color='white')),
                                  row=row,
                                  col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value = max([
            (day_values[i] if day_values[i] is not None else 0) +
            (night_values[i] if night_values[i] is not None else 0)
            for i in range(len(cities_ordered))
        ]) if cities_ordered else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_cities_per_col * 2  # The last row in the left column
        last_row_right_column = (len(cities_ordered) - num_cities_per_col) * 2  # The last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_cities_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column
            if i % 2 == 1:  # Odd rows
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=False
                )

            # Update x-axis for the right column
            if i % 2 == 1:  # Odd rows
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=False
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(
            title=dict(text=title_text,
                       font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=1
        )

        fig.update_xaxes(
            title=dict(text=title_text,
                       font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=2
        )

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='stack',
            height=TALL_FIG_HEIGHT,
            width=2480,
            showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150),
            bargap=0,
            bargroupgap=0
        )

        # Define gridline generation parameters
        if metric == "speed":
            start, step, count = 1, 1, 19
        elif metric == "time":
            start, step, count = 2, 2, 26

        # Generate gridline positions
        x_grid_values = [start + i * step for i in range(count)]

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x,
                y0=0,
                x1=x,
                y1=1,  # Set the position of the gridlines
                xref='x',
                yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x,
                y0=0,
                x1=x,
                y1=1,  # Set the position of the gridlines
                xref='x2',
                yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        if data_view == "combined":
            # Define the legend items
            legend_items = [
                {"name": "Day", "color": bar_colour_1},
                {"name": "Night", "color": bar_colour_2},
            ]

            # Add the vertical legends at the top and bottom
            self.add_vertical_legend_annotations(fig,
                                                 legend_items,
                                                 x_position=legend_x,
                                                 y_start=legend_y,
                                                 spacing=legend_spacing,
                                                 font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=1,
            x1=0.495,
            y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0.505,
            y0=1,
            x1=1,
            y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso']  # type: ignore
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        if order_by == "alphabetical":
            # Split cities into left and right columns
            left_column_cities = cities_ordered[:num_cities_per_col]
            right_column_cities = cities_ordered[num_cities_per_col:]

            # Adjust x positioning for the left and right columns
            x_position_left = 0.0  # Position for the left column
            x_position_right = 1.0  # Position for the right column
            font_size = FLAG_SIZE  # Font size for visibility

            # Initialise variables for dynamic y positioning for both columns
            current_row_left = 1  # Start from the first row for the left column
            current_row_right = 1  # Start from the first row for the right column
            y_position_map_left = {}  # Store y positions for each country (left column)
            y_position_map_right = {}  # Store y positions for each country (right column)

            # Calculate the y positions dynamically for the left column
            for city in left_column_cities:
                country = final_dict[city]['iso']

                if country not in y_position_map_left:  # Add the country label once per country
                    y_position_map_left[country] = 1 - (current_row_left - 1) / ((len(left_column_cities)-1.12) * 2)

                current_row_left += 2  # Increment the row for each city (speed and time take two rows)

            # Calculate the y positions dynamically for the right column
            for city in right_column_cities:
                country = final_dict[city]['iso']

                if country not in y_position_map_right:  # Add the country label once per country
                    y_position_map_right[country] = 1 - (current_row_right - 1) / ((len(right_column_cities)-1.12) * 2)

                current_row_right += 2  # Increment the row for each city (speed and time take two rows)

            # Add annotations for country names dynamically for the left column
            for country, y_position in y_position_map_left.items():
                iso2 = wrapper_class.iso3_to_iso2(country)
                country = country + wrapper_class.iso2_to_flag(iso2)
                fig.add_annotation(
                    x=x_position_left,  # Left column x position
                    y=y_position,  # Calculated y position based on the city order
                    xref="paper",
                    yref="paper",
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
                iso2 = wrapper_class.iso3_to_iso2(country)
                country = country + wrapper_class.iso2_to_flag(iso2)
                fig.add_annotation(
                    x=x_position_right,  # Right column x position
                    y=y_position,  # Calculated y position based on the city order
                    xref="paper",
                    yref="paper",
                    text=country,  # Country name
                    showarrow=False,
                    font=dict(size=font_size, color="black"),
                    xanchor='left',
                    align='left',
                    bgcolor='rgba(255,255,255,0.8)',  # Background color for visibility
                    # bordercolor="black",  # Border for visibility
                )

        fig.update_yaxes(
            tickfont=dict(size=TEXT_SIZE, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=80, r=80, t=x_axis_title_height, b=10))
        self.save_plotly_figure(fig=fig,
                                filename=filename,
                                width=2400,
                                height=TALL_FIG_HEIGHT,
                                scale=SCALE,
                                save_eps=False,
                                save_final=True)

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
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
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
                city_state = wrapper_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")

            logger.info("Top 5 cities with min |speed at day - speed at night| differences:")
            for city, diff in top_5_min_speed:
                city_state = wrapper_class.process_city_string(city, df_mapping)
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
                city_state = wrapper_class.process_city_string(city, df_mapping)
                logger.info(f"{city_state}: {diff}")

            logger.info("Top 5 cities with min |time_0 - time_1| differences:")
            for city, diff in top_5_min:
                city_state = wrapper_class.process_city_string(city, df_mapping)
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

            logger.info(f"City with max speed at day: {wrapper_class.process_city_string(max_speed_city_0, df_mapping)} with speed of {max_speed_value_0} m/s")  # noqa:E501
            logger.info(f"City with min speed at day: {wrapper_class.process_city_string(min_speed_city_0, df_mapping)} with speed of {min_speed_value_0} m/s")  # noqa:E501

        if filtered_dict_s_1:
            max_speed_city_1 = max(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            min_speed_city_1 = min(filtered_dict_s_1, key=lambda city: filtered_dict_s_1[city]["speed_1"])
            max_speed_value_1 = filtered_dict_s_1[max_speed_city_1]["speed_1"]
            min_speed_value_1 = filtered_dict_s_1[min_speed_city_1]["speed_1"]

            logger.info(f"City with max speed at night: {wrapper_class.process_city_string(max_speed_city_1, df_mapping)} with speed of {max_speed_value_1} m/s")  # noqa:E501
            logger.info(f"City with min speed at night: {wrapper_class.process_city_string(min_speed_city_1, df_mapping)} with speed of {min_speed_value_1} m/s")  # noqa:E501

        # Find city with max and min time_0 and time_1
        if filtered_dict_t_0:
            max_time_city_0 = max(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            min_time_city_0 = min(filtered_dict_t_0, key=lambda city: filtered_dict_t_0[city]["time_0"])
            max_time_value_0 = filtered_dict_t_0[max_time_city_0]["time_0"]
            min_time_value_0 = filtered_dict_t_0[min_time_city_0]["time_0"]

            logger.info(f"City with max time at day: {wrapper_class.process_city_string(max_time_city_0, df_mapping)} with time of {max_time_value_0} s")  # noqa:E501
            logger.info(f"City with min time at day: {wrapper_class.process_city_string(min_time_city_0, df_mapping)} with time of {min_time_value_0} s")  # noqa:E501

        if filtered_dict_t_1:
            max_time_city_1 = max(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            min_time_city_1 = min(filtered_dict_t_1, key=lambda city: filtered_dict_t_1[city]["time_1"])
            max_time_value_1 = filtered_dict_t_1[max_time_city_1]["time_1"]
            min_time_value_1 = filtered_dict_t_1[min_time_city_1]["time_1"]

            logger.info(f"City with max time at night: {wrapper_class.process_city_string(max_time_city_1, df_mapping)} with time of {max_time_value_1} s")  # noqa:E501
            logger.info(f"City with min time at night: {wrapper_class.process_city_string(min_time_city_1, df_mapping)} with time of {min_time_value_1} s")  # noqa:E501

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
        TALL_FIG_HEIGHT = num_cities_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col * 2, cols=2,  # Two columns
            vertical_spacing=0,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[2.0] * (num_cities_per_col * 2),
        )

        # Plot left column (first half of cities)
        for i, city in enumerate(cities_ordered[:num_cities_per_col]):
            city = wrapper_class.process_city_string(city, df_mapping)

            # Row for speed (Day and Night)
            row = 2 * i + 1
            if day_avg_speed[i] is not None and night_avg_speed[i] is not None:
                value = (day_avg_speed[i] + night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_avg_speed[i] is not None:  # Only day data available
                value = (day_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_avg_speed[i] is not None:  # Only night data available
                value = (night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night",
                    marker=dict(color=bar_colour_2), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            # Row for time (Day and Night)
            row = 2 * i + 2
            if day_time_dict[i] is not None and night_time_dict[i] is not None:
                value = (day_time_dict[i] + night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_4), text=[''],
                    textposition='auto', showlegend=False), row=row, col=1)

            elif day_time_dict[i] is not None:  # Only day time data available
                value = (day_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_time_dict[i] is not None:  # Only night time data available
                value = (night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_4),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

        # Similarly for the right column
        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city = wrapper_class.process_city_string(city, df_mapping)

            row = 2 * i + 1
            idx = num_cities_per_col + i
            if day_avg_speed[idx] is not None and night_avg_speed[idx] is not None:
                value = (day_avg_speed[idx] + night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_avg_speed[idx] is not None:
                value = (day_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_avg_speed[idx] is not None:
                value = (night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            row = 2 * i + 2
            if day_time_dict[idx] is not None and night_time_dict[idx] is not None:
                value = (day_time_dict[idx] + night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_4), text=[''],
                    textposition='inside', showlegend=False), row=row, col=2)

            elif day_time_dict[idx] is not None:  # Only day time data available
                value = (day_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_time_dict[idx] is not None:  # Only night time data available
                value = (night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{city} {value:.2f}'], orientation='h',
                    name=f"{city} time during night", marker=dict(color=bar_colour_4),
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
            {"name": "Crossing speed during daytime", "color": bar_colour_1},
            {"name": "Crossing speed during night time", "color": bar_colour_2},
            {"name": "Crossing decision time during daytime", "color": bar_colour_3},
            {"name": "Crossing decision time during night time", "color": bar_colour_4},
        ]

        # Add the vertical legends at the top and bottom
        self.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
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
            iso2 = wrapper_class.iso3_to_iso2(country)
            country = country + wrapper_class.iso2_to_flag(iso2)
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
            iso2 = wrapper_class.iso3_to_iso2(country)
            country = country + wrapper_class.iso2_to_flag(iso2)
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

        self.save_plotly_figure(fig,
                                "consolidated",
                                height=TALL_FIG_HEIGHT*2,
                                width=4960,
                                scale=SCALE,
                                save_final=True,
                                save_eps=False,
                                save_png=False)

    def mapbox_map(self, df, density_col=None, density_radius=30, hover_data=None, file_name="mapbox_map",
                   save_final=True):
        """Generate world map with cities using mapbox.

        Args:
            df (dataframe): dataframe with mapping info.
            hover_data (list, optional): list of params to show on hover.
            file_name (str, optional): name of file
        """
        # Draw map without density layer
        if not density_col:
            fig = px.scatter_map(df,
                                 lat="lat",
                                 lon="lon",
                                 hover_data=hover_data,
                                 hover_name="city",
                                 color=df["continent"],
                                 zoom=1.3)  # type: ignore
        # Draw map with density layer
        else:
            fig = px.density_mapbox(
                df,
                lat="lat",
                lon="lon",
                z=density_col,
                radius=density_radius,  # tune for spread
                zoom=2.5,  # type: ignore
                center=dict(lat=df["lat"].mean(), lon=df["lon"].mean()),
                mapbox_style="carto-positron",
                hover_name="city",
                hover_data=hover_data,
            )

            # fig.update_layout(
            #     height=700
            # )

        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            mapbox=dict(zoom=1.3),
            font=dict(family=common.get_configs('font_family'),
                      size=common.get_configs('font_size')),
            legend=dict(
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.6)',  # transparent white
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ),
            legend_title_text=""  # remove legend title
        )

        # Save and display the figure
        self.save_plotly_figure(fig, file_name, save_final=True)

    def world_map(self, df_mapping):
        """
        Generate a world map with highlighted countries and red markers for cities using Plotly.

        - Highlights countries based on the cities present in the dataset.
        - Adds scatter points for each city with detailed socioeconomic and traffic-related hover info.
        - Adjusts map appearance to improve clarity and remove irrelevant regions like Antarctica.

        Args:
            df_mapping (pd.DataFrame): A DataFrame with columns:
                ['city', 'state', 'country', 'lat', 'lon', 'continent',
                 'gmp', 'population_city', 'population_country',
                 'traffic_mortality', 'literacy_rate', 'avg_height',
                 'gini', 'traffic_index']

        Returns:
            None. Saves and displays the interactive map to disk.
        """
        cities = df_mapping["city"]
        states = df_mapping["state"]
        countries = df_mapping["country"]
        coords_lat = df_mapping["lat"]
        coords_lon = df_mapping["lon"]

        # Create the country list to highlight in the choropleth map
        countries_set = set(countries)  # Use set to avoid duplicates
        # if "Denmark" in countries_set:
        #     countries_set.add('Greenland')
        if "Türkiye" in countries_set:
            countries_set.add('Turkey')
        if "Somalia" in countries_set:
            countries_set.add('Somaliland')

        # Create a DataFrame for highlighted countries with a value (same for all to have the same color)
        df = pd.DataFrame({'country': list(countries_set), 'value': 1})

        # Create a choropleth map using Plotly with grey color for countries
        fig = px.choropleth(df, locations="country", locationmode="country names",
                            color="value", hover_name="country", hover_data={'value': False, 'country': False},
                            color_continuous_scale=["rgb(242, 186, 78)", "rgb(242, 186, 78)"],
                            labels={'value': 'Highlighted'})

        # Update layout to remove Antarctica, Easter Island, remove the color bar, and set ocean color
        fig.update_layout(
            coloraxis_showscale=False,  # Remove color bar
            geo=dict(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="black",  # Set coastline color
                showcountries=True,  # Show country borders
                countrycolor="black",  # Set border color
                projection_type='equirectangular',
                showlakes=True,
                lakecolor='rgb(173, 216, 230)',  # Light blue for lakes
                projection_scale=1,
                center=dict(lat=20, lon=0),  # Center map to remove Antarctica
                bgcolor='rgb(173, 216, 230)',  # Light blue for ocean
                resolution=50
            ),
            margin=dict(l=0, r=0, t=0, b=0),  # Remove the margins
            paper_bgcolor='rgb(173, 216, 230)'  # Set the paper background to match the ocean color
        )

        # Process each city and its corresponding country
        city_coords = []
        for i, (city, state, lat, lon) in enumerate(tqdm(zip(cities, states, coords_lat, coords_lon), total=len(cities))):  # noqa: E501
            if not state or str(state).lower() == 'nan':
                state = 'N/A'
            if lat and lon:
                city_coords.append({
                    'City': city,
                    'State': state,
                    'Country': df_mapping["country"].iloc[i],
                    'Continent': df_mapping["continent"].iloc[i],
                    'lat': lat,
                    'lon': lon,
                    'GDP (Billion USD)': df_mapping["gmp"].iloc[i],
                    'City population (thousands)': df_mapping["population_city"].iloc[i] / 1000.0,
                    'Country population (thousands)': df_mapping["population_country"].iloc[i] / 1000.0,
                    'Traffic mortality rate (per 100,000)': df_mapping["traffic_mortality"].iloc[i],
                    'Literacy rate': df_mapping["literacy_rate"].iloc[i],
                    'Average height (cm)': df_mapping["avg_height"].iloc[i],
                    'Gini coefficient': df_mapping["gini"].iloc[i],
                    'Traffic index': df_mapping["traffic_index"].iloc[i],
                })

        if city_coords:
            city_df = pd.DataFrame(city_coords)
            # city_df["City"] = city_df["city"]  # Format city name with "City:"
            city_trace = px.scatter_geo(
                city_df, lat='lat', lon='lon',
                hover_data={
                    'City': True,
                    'State': True,
                    'Country': True,
                    'Continent': True,
                    'GDP (Billion USD)': True,
                    'City population (thousands)': True,
                    'Country population (thousands)': True,
                    'Traffic mortality rate (per 100,000)': True,
                    'Literacy rate': True,
                    'Average height (cm)': True,
                    'Gini coefficient': True,
                    'Traffic index': True,
                    'lat': False,
                    'lon': False  # Hide lat and lon
                }
            )
            # Update the city markers to be red and adjust size
            city_trace.update_traces(marker=dict(color="red", size=5))

            # Add the scatter_geo trace to the choropleth map
            fig.add_trace(city_trace.data[0])

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Save and display the figure
        self.save_plotly_figure(fig, "world_map", save_final=True)

    def hist(self, data_index, name, min_threshold, max_threshold, nbins=None, color=None,
             pretty_text=False, marginal='rug', xaxis_title=None, yaxis_title=None,
             name_file=None, save_file=False, save_final=False, fig_save_width=1320,
             fig_save_height=680, font_family=None, font_size=None, vlines=None, xrange=None):
        """
        Output histogram of selected data from pickle file.

        Args:
            data_index (int): index of the item in the tuple to plot.
            nbins (int, optional): number of bins in histogram.
            color (str, optional): dataframe column to assign colour of bars.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising.
            marginal (str, optional): marginal type: 'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): whether to save HTML file of the plot.
            save_final (bool, optional): whether to save final figure to /figures.
            fig_save_width (int, optional): width of saved figure.
            fig_save_height (int, optional): height of saved figure.
            font_family (str, optional): font family to use. Defaults to config.
            font_size (int, optional): font size to use. Defaults to config.
        """

        # Load data from pickle file
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)
        nested_dict = data_tuple[data_index]

        all_values = [speed for city in nested_dict.values() for video in city.values() for speed in video.values()]
        all_values = [value for value in all_values
                      if (min_threshold is None or value >= min_threshold)
                      and (max_threshold is None or value <= max_threshold)]

        # --- Calculate mean and median ---
        mean_val = np.mean(all_values)
        median_val = np.median(all_values)

        logger.info('Creating histogram for {}.', name)

        # Restrict values to the specified x-range if provided
        if xrange is not None:
            x_min, x_max = xrange
            all_values = [x for x in all_values if x_min <= x <= x_max]

        # Create histogram
        if color:
            fig = px.histogram(x=all_values, nbins=nbins, marginal=marginal, color=color)
        else:
            fig = px.histogram(x=all_values, nbins=nbins, marginal=marginal)

        fig.update_layout(
            xaxis=dict(tickformat='digits'),
            template=common.get_configs('plotly_template'),
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            font=dict(
                family=font_family if font_family else common.get_configs('font_family'),
                size=font_size if font_size else common.get_configs('font_size')
            )
        )

        # --- Add vertical lines for mean and median ---
        fig.add_vline(
            x=mean_val,
            line_dash='dash',
            line_color='blue',
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position='top right'
        )

        fig.add_vline(
            x=median_val,
            line_dash='dash',
            line_color='red',
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position='top left'
        )

        if vlines:
            for x in vlines:
                fig.add_vline(
                    x=x,
                    line_dash='dot',
                    line_color='black',
                    annotation_text=f'{x}',
                    annotation_position='top'
                )

        if save_file:
            if not name_file:
                name_file = f"hist_{name}"
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            self.save_plotly_figure(fig, name_file, save_final=True)
        else:
            fig.show()

    def violin_plot(self, data_index, name, min_threshold, max_threshold, df_mapping, color=None,
                    pretty_text=False, xaxis_title=None, yaxis_title=None,
                    name_file=None, save_file=False, save_final=False, fig_save_width=1320,
                    fig_save_height=680, font_family=None, font_size=None, vlines=None, xrange=None):

        # Load data from pickle file
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)
        nested_dict = data_tuple[data_index]

        values = {}

        for city_lat_long_cond, inner_dict in nested_dict.items():
            city, lat, _, _ = city_lat_long_cond.split("_")
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            if country not in values:
                values[country] = []  # Use a list to keep duplicates
            for video_id, inner_values in inner_dict.items():
                values[country].extend(inner_values.values())

        fig = go.Figure()

        # Add one violin plot per city
        for country, country_values in values.items():
            fig.add_trace(go.Violin(
                y=country_values,
                x=[country] * len(country_values),  # Repeating city name for each value
                name=country,
                box_visible=True,
                meanline_visible=True,
                # points='all'  # Optional: show individual points
            ))

        # Update layout to improve visualization
        fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Values",
            violinmode='group'
        )

        fig.show()

    def map(self, df, color, title, title_colorbar=None, save_file=False):
        """Map of countries of participation with colour based on column in dataframe.

        Args:
            df (dataframe): dataframe with keypress data.
        """
        logger.info('Creating visualisation of heatmap of participants by country with colour defined by {}.', color)

        # Filter out rows where the color value is 0 or NaN
        df_filtered = df[df[color].fillna(0) != 0].copy()

        # Get Denmark's value for the specified column
        denmark_value = df_filtered.loc[df_filtered['country'] == 'Denmark', color].values
        if len(denmark_value) > 0:
            greenland_row = {
                'country': 'Greenland',
                color: denmark_value[0]
            }
            # Add any other required columns with default or NaN
            for col in df_filtered.columns:
                if col not in greenland_row:
                    greenland_row[col] = None

            df_filtered = pd.concat([df_filtered, pd.DataFrame([greenland_row])], ignore_index=True)

        # ---- HANDLE COUNTRY NAME MISMATCHES ----
        country_name_map = {
            'Türkiye': 'Turkey'
        }
        df_filtered['country'] = df_filtered['country'].replace(country_name_map)
        # ----------------------------------------

        # create map
        fig = px.choropleth(df_filtered,
                            locations='country',
                            locationmode='country names',
                            color=color,
                            hover_name='country',
                            color_continuous_scale=px.colors.sequential.Plasma)
        fig.update_layout(
            font=dict(
                family=common.get_configs('font_family'),
                size=common.get_configs('font_size')
            ),
            coloraxis_colorbar=dict(
                x=0,              # far left
                xanchor='left',
                y=0.45,            # vertically centered
                len=0.7,         # adjust the length of the color bar
                thickness=20,     # make it thinner if needed
                title=title_colorbar     # optional: title for clarity
            )
        )
        # save file to local output folder
        if save_file:
            # Final adjustments and display
            fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
            self.save_plotly_figure(fig, f"map_{color}", save_final=True)
        # open it in localhost instead
        else:
            fig.show()

    def map_political(self, df, df_mapping, show_images=False, show_cities=True, hover_data=None, save_file=False,
                      save_final=False):
        """Generate world map with countries colored by continent using choropleth.

        Args:
            df (dataframe): dataframe with 'country' and 'continent' columns.
            hover_data (list, optional): list of params to show on hover.
        """
        if 'Denmark' in df['country'].values:
            denmark_value = df.loc[df['country'] == 'Denmark', 'continent'].values[0]
            df = pd.concat([df, pd.DataFrame([{'country': 'Greenland', 'continent': denmark_value}])],
                           ignore_index=True)

        country_name_map = {
            "Türkiye": "Turkey"
        }

        # Replace the names in your DataFrame
        df['country'] = df['country'].replace(country_name_map)

        # Plot country-level choropleth
        fig = px.choropleth(df,
                            locations="country",
                            locationmode="country names",
                            color="continent",
                            hover_name="country",
                            hover_data=hover_data,
                            projection="natural earth")

        # add markers of cities
        if show_cities:
            # add city markers as scattergeo
            fig.add_trace(go.Scattergeo(
                lon=df_mapping['lon'],
                lat=df_mapping['lat'],
                text=df_mapping.get('city', None),
                mode='markers',
                hoverinfo='skip',
                marker=dict(
                    size=4,
                    color='black',
                    opacity=0.7,
                    symbol='circle'
                ),
                name='cities'
            ))

        # add screenshots of videos
        if show_images:
            # define city images with positions
            city_images = [
                {
                    "city": "Tokyo",
                    "file": "tokyo.png",
                    "x": 0.933, "y": 0.58,
                    "approx_lon": 165.2, "approx_lat": 7.2,
                    "label": "Tokyo, Japan",
                    "x_label": 0.983, "y_label": 0.641,
                    "video": "oDejyTLYUTE",
                    "x_video": 0.933-0.0021, "y_video": 0.58-0.059
                },
                {
                    "city": "Nairobi",
                    "file": "nairobi.png",
                    "x": 0.72, "y": 0.38,
                    "approx_lon": 70.2, "approx_lat": -20.0,
                    "label": "Nairobi, Kenya",
                    "x_label": 0.7695, "y_label": 0.38+0.062,
                    "video": "VNLqnwoJqmM",
                    "x_video": 0.72+0.00529, "y_video": 0.38-0.069,
                },
                {
                    "city": "Los Angeles",
                    "file": "los_angeles.png",
                    "x": 0.12, "y": 0.5,
                    "approx_lon": -121.7, "approx_lat": 0.0,
                    "label": "Los Angeles, CA, USA",
                    "x_label": 0.07, "y_label": 0.5+0.062,
                    "video": "4uhMg5na888",
                    "x_video": 0.12-0.002, "y_video": 0.5-0.06,
                },
                {
                    "city": "Paris",
                    "file": "paris.png",
                    "x": 0.3915, "y": 0.68,
                    "approx_lon": -30.6, "approx_lat": 30.4,
                    "label": "Paris, France",
                    "x_label": 0.37, "y_label": 0.68+0.072,
                    "video": "ZTmjk8mSCq8",
                    "x_video": 0.3915-0.0225, "y_video": 0.68-0.06,
                },
                {
                    "city": "Rio de Janeiro",
                    "file": "rio_de_janeiro.png",
                    "x": 0.47, "y": 0.2,
                    "approx_lon": -1.8, "approx_lat": -60.2,
                    "label": "Rio de Janeiro, Brazil",
                    "x_label": 0.4746, "y_label": 0.2+0.05,
                    "video": "q83bl_GcsCo",
                    "x_video": 0.47-0.026, "y_video": 0.2-0.069,
                },
                {
                    "city": "Melbourne",
                    "file": "melbourne.png",
                    "x": 0.74, "y": 0.22,
                    "approx_lon": 90.0, "approx_lat": -52.0,
                    "label": "Melbourne, Australia",
                    "x_label": 0.7783, "y_label": 0.22+0.05,
                    "video": "gQ-9mmnfJjE",
                    "x_video": 0.74, "y_video": 0.22-0.069,
                }
            ]

            path_screenshots = os.path.join(common.root_dir, 'readme')
            # add each image
            for item in city_images:
                fig.add_layout_image(
                    dict(
                        source=os.path.join(path_screenshots, item['file']),
                        xref="paper", yref="paper",
                        x=item["x"], y=item["y"],
                        sizex=0.1, sizey=0.1,
                        xanchor="center", yanchor="middle",
                        layer="above"
                    )
                )
                # text label on top
                if "label" in item:
                    fig.add_annotation(
                        text=item["label"],
                        x=item["x_label"],
                        y=item["y_label"],
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black",
                        borderwidth=1
                    )

            # draw arrows from image to city location
            for item in city_images:
                row = df_mapping[df_mapping['city'].str.lower() == item['city'].lower()]
                if not row.empty:
                    fig.add_trace(go.Scattergeo(
                        lon=[item['approx_lon'], row['lon'].values[0]],
                        lat=[item['approx_lat'], row['lat'].values[0]],
                        mode='lines',
                        line=dict(width=2, color='black'),
                        showlegend=False,
                        geo='geo',
                        hoverinfo='skip'
                    ))
                    # label with video on the bottom
                    fig.add_annotation(
                        dict(
                            text=item['video'],
                            x=item["x_video"], y=item["y_video"],
                            xref="paper", yref="paper",
                            showarrow=False,
                            font=dict(size=10, color="black"),
                            align="center",
                            bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="black",
                            borderwidth=1
                        )
                    )

            # add YOLO image
            fig.add_layout_image(
                dict(
                    source=os.path.join(path_screenshots, 'new_york_yolo.png'),  # or use PIL.Image.open if needed
                    xref="paper", yref="paper",
                    x=0.2, y=0.25,
                    sizex=0.2, sizey=0.2,
                    xanchor="center", yanchor="middle",
                    layer="above"
                )
            )
            # label on top
            fig.add_annotation(
                dict(
                    text="Example of YOLO output (New York, NY, USA)",
                    x=0.1001, y=0.25+0.1115,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center",
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="black",
                    borderwidth=1
                )
            )
            # label with video on the bottom
            # text label on top
            fig.add_annotation(
                dict(
                    text="Wyg213IZDI",
                    x=0.253, y=0.25-0.119,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    align="center",
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="black",
                    borderwidth=1
                )
            )

        # Remove color bar
        fig.update_coloraxes(showscale=False)

        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            font=dict(
                family=common.get_configs('font_family'),
                size=common.get_configs('font_size')
            )
        )

        # save file to local output folder
        if save_file:
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            # with screenshots
            if show_images:
                self.save_plotly_figure(fig, "map_screenshots", save_final=False)
            # without screenshots
            else:
                self.save_plotly_figure(fig, "map", save_final=True)
        # open it in localhost instead
        else:
            fig.show()

    def scatter(self, df, x, y, extension=None, color=None, symbol=None, size=None, text=None, trendline=None,
                hover_data=None, marker_size=None, pretty_text=False, marginal_x='violin', marginal_y='violin',
                xaxis_title=None, yaxis_title=None, xaxis_range=None, yaxis_range=None, name_file=None,
                save_file=False, save_final=False, fig_save_width=1320, fig_save_height=680, font_family=None,
                font_size=None, hover_name=None, legend_title=None, legend_x=None, legend_y=None,
                label_distance_factor=1.0):
        """
        Output scatter plot of variables x and y with optional assignment of colour and size.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (str): dataframe column to plot on x axis.
            y (str): dataframe column to plot on y axis.
            color (str, optional): dataframe column to assign colour of points.
            symbol (str, optional): dataframe column to assign symbol of points.
            size (str, optional): dataframe column to assign doze of points.
            text (str, optional): dataframe column to assign text labels.
            trendline (str, optional): trendline. Can be 'ols', 'lowess'
            hover_data (list, optional): dataframe columns to show on hover.
            marker_size (int, optional): size of marker. Should not be used together with size argument.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            marginal_x (str, optional): type of marginal on x axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
            hover_name (list, optional): title on top of hover popup.
            legend_title (list, optional): title on top of legend.
            legend_x (float, optional): x position of legend.
            legend_y (float, optional): y position of legend.
            label_distance_factor (float, optional): multiplier for the threshold to control density of text labels.
        """
        logger.info('Creating scatter plot for x={} and y={}.', x, y)
        # using size and marker_size is not supported
        if marker_size and size:
            logger.error('Arguments marker_size and size cannot be used together.')
            return -1
        # using marker_size with histogram marginal(s) is not supported
        if (marker_size and (marginal_x == 'histogram' or marginal_y == 'histogram')):
            logger.error('Argument marker_size cannot be used together with histogram marginal(s).')
            return -1
        # prettify text
        if pretty_text:
            if isinstance(df.iloc[0][x], str):  # check if string
                # replace underscores with spaces
                df[x] = df[x].str.replace('_', ' ')
                # capitalise
                df[x] = df[x].str.capitalize()
            if isinstance(df.iloc[0][y], str):  # check if string
                # replace underscores with spaces
                df[y] = df[y].str.replace('_', ' ')
                # capitalise
                df[y] = df[y].str.capitalize()
            if color and isinstance(df.iloc[0][color], str):  # check if string
                # replace underscores with spaces
                df[color] = df[color].str.replace('_', ' ')
                # capitalise
                df[color] = df[color].str.capitalize()
            if size and isinstance(df.iloc[0][size], str):  # check if string
                # replace underscores with spaces
                df[size] = df[size].str.replace('_', ' ')
                # capitalise
                df[size] = df[size].str.capitalize()
            try:
                # check if string
                if text and isinstance(df.iloc[0][text], str):
                    # replace underscores with spaces
                    df[text] = df[text].str.replace('_', ' ')
                    # capitalise
                    df[text] = df[text].str.capitalize()
            except ValueError as e:
                logger.debug('Tried to prettify {} with exception {}.', text, e)

        # check and clean the data
        df = df.replace([np.inf, -np.inf], np.nan).dropna()  # Remove NaNs and Infs

        if text:
            if text in df.columns:
                # use KDTree to check point density
                tree = KDTree(df[[x, y]].values)  # Ensure finite values
                distances, _ = tree.query(df[[x, y]].values, k=2)  # Find nearest neighbor distance

                # define a distance threshold for labeling
                threshold = np.mean(distances[:, 1]) * label_distance_factor

                # only label points that are not too close to others
                df["display_label"] = np.where(distances[:, 1] > threshold, df[text], "")

                text = "display_label"
            else:
                logger.warning("Column 'country' not found, skipping display_label logic.")

        # scatter plot with histograms
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            fig = px.scatter(df,
                             x=x,
                             y=y,
                             color=color,
                             symbol=symbol,
                             size=size,
                             text=text,
                             trendline=trendline,
                             hover_data=hover_data,
                             hover_name=hover_name,
                             marginal_x=marginal_x,
                             marginal_y=marginal_y)

        # font size of text labels
        for trace in fig.data:
            if trace.type == "scatter" and "text" in trace:  # type: ignore
                trace.textfont = dict(size=common.get_configs('font_size'))  # type: ignore

        # location of labels
        if not marginal_x and not marginal_y:
            fig.update_traces(textposition=Plots.improve_text_position(df[x]))

        # update layout
        fig.update_layout(template=common.get_configs('plotly_template'),
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)

        # change marker size
        if marker_size:
            fig.update_traces(marker=dict(size=marker_size))

        # update legend title
        if legend_title is not None:
            fig.update_layout(legend_title_text=legend_title)

        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=common.get_configs('font_size')))

        # legend
        if legend_x and legend_y:
            fig.update_layout(legend=dict(x=legend_x, y=legend_y, bgcolor='rgba(0,0,0,0)'))

        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                if extension is not None:
                    name_file = 'scatter_' + x + '-' + y + '-' + extension
                else:
                    name_file = 'scatter_' + x + '-' + y

            # Final adjustments and display
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            self.save_plotly_figure(fig, name_file, save_final=True)
        # open it in localhost instead
        else:
            fig.show()

    @staticmethod
    def improve_text_position(x):
        """
        Generate a list of text positions for plotting annotations based on the length of the input list `x`.

        This function alternates between predefined text positions (e.g., 'top center', 'bottom center')
        for each item in `x`. It is more efficient and visually clear if the corresponding x-values are sorted
        before using this function.

        Args:
            x (list): A list of values (typically x-axis data points).

        Returns:
            list: A list of text positions corresponding to each element in `x`.
        """
        # Predefined positions to alternate between (can be extended with more options)
        positions = ['top center', 'bottom center']

        # Cycle through the positions for each element in the input list
        return [positions[i % len(positions)] for i in range(len(x))]

    def stack_plot_country(self, df_mapping, order_by, metric, data_view, title_text, filename,
                           legend_x=0.87, legend_y=0.04, font_size_captions=40,
                           legend_spacing=0.02, left_margin=10, right_margin=10, top_margin=0,
                           bottom_margin=0):
        """
        Plots a stacked bar graph based on the provided data and configuration.

        Parameters:
            df_mapping (dict): A dictionary mapping categories to their respective DataFrames.
            order_by (str): Criterion to order the bars, e.g., 'alphabetical' or 'average'.
            metric (str): The metric to visualize, such as 'speed' or 'time'.
            data_view (str): Determines which subset of data to visualise, such as 'day', 'night', or 'combined'.
            title_text (str): The title of the plot.
            filename (str): The name of the file to save the plot as.
            font_size_captions (int, optional): Font size for captions. Default is 40.
            x_axis_title_height (int, optional): Vertical space for x-axis title. Default is 110.
            legend_x (float, optional): X position of the legend. Default is 0.92.
            legend_y (float, optional): Y position of the legend. Default is 0.015.
            legend_spacing (float, optional): Spacing between legend entries. Default is 0.02.

        Returns:
            None
        """

        # Define log messages in a structured way
        log_messages = {
            ("alphabetical", "speed", "day"): "Plotting speed to cross by alphabetical order during day time.",
            ("alphabetical", "speed", "night"): "Plotting speed to cross by alphabetical order during night time.",
            ("alphabetical", "speed", "combined"): "Plotting speed to cross by alphabetical order.",
            ("alphabetical", "time", "day"): "Plotting time to start cross by alphabetical order during day time.",
            ("alphabetical", "time", "night"): "Plotting time to start cross by alphabetical order during night time.",
            ("alphabetical", "time", "combined"): "Plotting time to start cross by alphabetical order.",
            ("average", "speed", "day"): "Plotting speed to cross by average during day time.",
            ("average", "speed", "night"): "Plotting speed to cross by averageduring night time.",
            ("average", "speed", "combined"): "Plotting speed to cross by average.",
            ("average", "time", "day"): "Plotting time to start cross by average during day time.",
            ("average", "time", "night"): "Plotting time to start cross by average during night time.",
            ("average", "time", "combined"): "Plotting time to start cross by average.",
            ("condition", "time", "combined"): "Plotting time to start cross sorted by day values.",
            ("condition", "speed", "combined"): "Plotting speed to cross sorted by day values."
        }

        message = log_messages.get((order_by, metric, data_view))
        final_dict = {}

        if message:
            logger.info(message)

        # Map metric names to their index in the data tuple
        metric_index_map = {
            "speed": 27,
            "time": 28,
            "all_speed_country": 38,
            "all_time_country": 39
            }

        if metric not in metric_index_map:
            raise ValueError(f"Unsupported metric: {metric}")

        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        metric_data = data_tuple[metric_index_map[metric]]
        all_values = data_tuple[metric_index_map[f"all_{metric}_country"]]

        if metric_data is None:
            raise ValueError(f"'{metric}' returned None, please check the input data or calculations.")

        # Clean NaNs
        metric_data = {
            key: value for key, value in metric_data.items()
            if not (isinstance(value, float) and math.isnan(value))
        }

        for country_condition, metric_values in tqdm(metric_data.items()):
            country, condition = country_condition.split('_')

            # Get the iso3 from the mapping file
            iso_code = values_class.get_value(df=df_mapping,
                                              column_name1="country",
                                              column_value1=country,
                                              column_name2=None,
                                              column_value2=None,
                                              target_column="iso3")

            if country is not None and iso_code is not None:
                # Initialise the country's dictionary if not already present
                if f'{country}' not in final_dict:
                    final_dict[f'{country}'] = {f"{metric}_0": None,
                                                f"{metric}_1": None,
                                                f"{metric}_sd_0": None,
                                                f"{metric}_sd_1": None,
                                                f"{metric}_sd_avg": None,
                                                "country": country,
                                                "iso3": iso_code}

                # Populate the corresponding speed based on the condition
                final_dict[f'{country}'][f"{metric}_{condition}"] = metric_values
                final_dict[f'{country}'][f"{metric}_sd_{condition}"] = np.std(all_values[f"{country}_{condition}"])

                vals = [v for k, v in all_values.items() if k.startswith(f"{country}_") and v is not None]
                flat_vals = list(itertools.chain.from_iterable(vals))
                final_dict[country][f"{metric}_sd_avg"] = np.std(flat_vals)

        if order_by == "alphabetical":
            if data_view == "day":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_0") or 0) >= 0.005
                    ],
                    key=lambda country: (final_dict[country].get("iso") or "")
                )
            elif data_view == "night":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_1") or 0) >= 0.005
                    ],
                    key=lambda country: (final_dict[country].get("iso") or "")
                )
            else:
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (((final_dict[country].get(f"{metric}_0") or 0) + (final_dict[country].get(f"{metric}_1") or 0)) / 2) >= 0.005  # noqa:E501
                    ],
                    key=lambda country: (final_dict[country].get("iso") or "")
                )

        elif order_by == "average":
            if data_view == "day":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_0") or 0) >= 0.005
                    ],
                    key=lambda country: final_dict[country].get(f"{metric}_0") or 0,
                    reverse=True
                )

            elif data_view == "night":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_1") or 0) >= 0.005
                    ],
                    key=lambda country: final_dict[country].get(f"{metric}_1") or 0,
                    reverse=True
                )

            else:
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (((final_dict[country].get(f"{metric}_0") or 0) + (final_dict[country].get(f"{metric}_1") or 0)) / 2) >= 0.005  # noqa:E501  # type: ignore
                    ],
                    key=lambda country: (
                        ((final_dict[country].get(f"{metric}_0") or 0) + (final_dict[country].get(f"{metric}_1") or 0)) / 2  # noqa:E501  # type: ignore
                    ), reverse=True
                )

        elif order_by == "condition":
            if data_view == "combined":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_0") is not None or final_dict[country].get(f"{metric}_1") is not None)  # noqa:E501
                        and ((final_dict[country].get(f"{metric}_0") or final_dict[country].get(f"{metric}_1") or 0) >= 0.005)  # noqa:E501
                    ],
                    key=lambda country: (
                        final_dict[country].get(f"{metric}_0")
                        if final_dict[country].get(f"{metric}_0") is not None
                        else final_dict[country].get(f"{metric}_1") or 0
                        ), reverse=True
                    )

        if len(countries_ordered) == 0:
            return

        # Prepare data for day and night stacking
        day_key = f"{metric}_0"
        night_key = f"{metric}_1"
        day_sd_key = f"{metric}_sd_0"
        night_sd_key = f"{metric}_sd_1"
        all_sd_key = f"{metric}_sd_avg"

        if data_view == "combined":
            day_values = [final_dict[country][day_key] for country in countries_ordered]
            night_values = [final_dict[country][night_key] for country in countries_ordered]
            day_sd = [final_dict[country][day_sd_key] for country in countries_ordered]
            night_sd = [final_dict[country][night_sd_key] for country in countries_ordered]
            all_sd = [final_dict[country][all_sd_key] for country in countries_ordered]

        elif data_view == "day":
            day_values = [final_dict[country][day_key] for country in countries_ordered]
            day_sd = [final_dict[country][day_sd_key] for country in countries_ordered]
            night_values = [0 for country in countries_ordered]
            night_sd = [0 for country in countries_ordered]

        elif data_view == "night":
            day_values = [0 for country in countries_ordered]
            day_sd = [0 for country in countries_ordered]
            night_values = [final_dict[country][night_key] for country in countries_ordered]
            night_sd = [final_dict[country][night_sd_key] for country in countries_ordered]

        # Determine how many countries will be in each column
        num_countries_per_col = len(countries_ordered) // 2 + len(countries_ordered) % 2  # Split cities

        # Define a base height per row and calculate total figure height
        TALL_FIG_HEIGHT = num_countries_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_countries_per_col, cols=2,  # Two columns
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_countries_per_col),
        )

        # Plot left column (first half of countries)
        for i, country in enumerate(countries_ordered[:num_countries_per_col]):

            # city = wrapper_class.process_city_string(city, df_mapping)
            iso_code = values_class.get_value(df_mapping, "country", country, None, None, "iso3")

            # build up textual label for left column
            country = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + country

            # Row for speed (Day and Night)
            row = i + 1
            if day_values[i] is not None and night_values[i] is not None:
                if data_view == "combined":
                    value = (day_values[i] + night_values[i])/2
                else:
                    value = (day_values[i] + night_values[i])

                # Determine the y value
                if order_by == "condition":
                    y_value = [
                        f"{country} {value:.2f}±{all_sd[i]:.2f} "
                        f"(D={day_values[i]:.2f}±{day_sd[i]:.2f}, "
                        f"N={night_values[i]:.2f}±{night_sd[i]:.2f})"]
                else:
                    y_value = [f'{country} {value:.2f}']

                fig.add_trace(go.Bar(
                    x=[day_values[i]],
                    y=y_value,
                    orientation='h',
                    marker=dict(color=bar_colour_1),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=18,
                                  color='white')),
                                  row=row,
                                  col=1)

                fig.add_trace(go.Bar(
                    x=[night_values[i]],
                    y=y_value,
                    orientation='h',
                    marker=dict(color=bar_colour_2),
                    text=[''],
                    textposition='inside',
                    showlegend=False),
                    row=row,
                    col=1)

            elif day_values[i] is not None:  # Only day data available
                value = day_values[i]

                # Determine the y value
                if order_by == "condition":
                    y_value = [f"{country} {value:.2f}±{day_sd[i]:.2f}"]
                else:
                    y_value = [f'{country} {value:.2f}']

                fig.add_trace(go.Bar(
                    x=[day_values[i]],
                    y=y_value,
                    orientation='h',
                    marker=dict(color=bar_colour_1),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=18,
                                  color='white')),
                                  row=row,
                                  col=1)

            elif night_values[i] is not None:  # Only night data available
                value = night_values[i]

                # Determine the y value
                if order_by == "condition":
                    y_value = [f"{country} {value:.2f}±{night_sd[i]:.2f}"]
                else:
                    y_value = [f'{country} {value:.2f}']

                fig.add_trace(go.Bar(
                    x=[night_values[i]],
                    y=y_value,
                    orientation='h',
                    marker=dict(color=bar_colour_2),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=18,
                                  color='white')),
                                  row=row,
                                  col=1)

        # Plot right column (second half of cities)
        for i, country in enumerate(countries_ordered[num_countries_per_col:]):
            # city = wrapper_class.process_city_string(city, df_mapping)
            iso_code = values_class.get_value(df_mapping, "country", country, None, None, "iso3")

            # build up textual label for left column
            country = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + country

            # Row for speed (Day and Night)
            row = i + 1
            idx = num_countries_per_col + i
            if day_values[idx] is not None and night_values[idx] is not None:
                if data_view == "combined":
                    value = (day_values[idx] + night_values[idx])/2
                else:
                    value = (day_values[idx] + night_values[idx])

                # Determine the y value
                if order_by == "condition":
                    y_value = [
                        f"{country} {value:.2f}±{all_sd[idx]:.2f} "
                        f"(D={day_values[idx]:.2f}±{day_sd[idx]:.2f}, "
                        f"N={night_values[idx]:.2f}±{night_sd[idx]:.2f})"]
                else:
                    y_value = [f'{country} {value:.2f}']

                fig.add_trace(go.Bar(
                    x=[day_values[idx]],
                    y=y_value,
                    orientation='h',
                    name=f"{country} {metric} during day",
                    marker=dict(color=bar_colour_1),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=14,
                                  color='white')),
                                  row=row,
                                  col=2)

                fig.add_trace(go.Bar(
                    x=[night_values[idx]],
                    y=y_value,
                    orientation='h',
                    name=f"{country} {metric} during night",
                    marker=dict(color=bar_colour_2),
                    text=[''],
                    textposition='inside',
                    showlegend=False),
                    row=row,
                    col=2)

            elif day_values[idx] is not None:
                value = day_values[idx]

                # Determine the y value
                if order_by == "condition":
                    y_value = [f"{country} {value:.2f}±{day_sd[idx]:.2f}"]
                else:
                    y_value = [f'{country} {value:.2f}']

                fig.add_trace(go.Bar(
                    x=[day_values[idx]],
                    y=y_value,
                    orientation='h',
                    name=f"{country} {metric} during day",
                    marker=dict(color=bar_colour_1),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=14,
                                  color='white')),
                                  row=row,
                                  col=2)

            elif night_values[idx] is not None:
                value = night_values[idx]

                # Determine the y value
                if order_by == "condition":
                    y_value = [f"{country} {value:.2f}±{night_sd[idx]:.2f}"]
                else:
                    y_value = [f'{country} {value:.2f}']

                fig.add_trace(go.Bar(
                    x=[night_values[idx]],
                    y=y_value,
                    orientation='h',
                    name=f"{country} {metric} during night",
                    marker=dict(color=bar_colour_2),
                    text=[''],
                    textposition='inside',
                    insidetextanchor='start',
                    showlegend=False,
                    textfont=dict(size=14,
                                  color='white')),
                                  row=row,
                                  col=2)

        # Calculate the maximum value across all data to use as x-axis range
        max_value = max([
            (day_values[i] if day_values[i] is not None else 0) +
            (night_values[i] if night_values[i] is not None else 0)
            for i in range(len(countries_ordered))
        ]) if countries_ordered else 0

        # Identify the last row for each column where the last city is plotted
        last_row_left_column = num_countries_per_col * 2  # The last row in the left column
        last_row_right_column = (len(countries_ordered) - num_countries_per_col) * 2  # Last row in the right column
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_countries_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column
            if i % 2 == 1:  # Odd rows
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=False
                )

            # Update x-axis for the right column
            if i % 2 == 1:  # Odd rows
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=False
                )

        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(
            title=dict(text=title_text,
                       font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=1
        )

        fig.update_xaxes(
            title=dict(text=title_text,
                       font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=2
        )

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='stack',
            height=2400,
            width=2480,
            showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150),
            bargap=0,
            bargroupgap=0
        )

        # Define gridline generation parameters
        if metric == "speed":
            start, step, count = 1, 1, 19
        elif metric == "time":
            start, step, count = 0.5, 0.5, 3

        # Generate gridline positions
        x_grid_values = [start + i * step for i in range(count)]

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x,
                y0=0,
                x1=x,
                y1=1,  # Set the position of the gridlines
                xref='x',
                yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x,
                y0=0,
                x1=x,
                y1=1,  # Set the position of the gridlines
                xref='x2',
                yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        if data_view == "combined":
            # Define the legend items
            legend_items = [
                {"name": "Day", "color": bar_colour_1},
                {"name": "Night", "color": bar_colour_2},
            ]

            # Add the vertical legends at the top and bottom
            self.add_vertical_legend_annotations(fig,
                                                 legend_items,
                                                 x_position=legend_x,
                                                 y_start=legend_y,
                                                 spacing=legend_spacing,
                                                 font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=1,
            x1=0.495,
            y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0.505,
            y0=1,
            x1=1,
            y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_city_map = {}
        for city, info in final_dict.items():
            country = info['iso3']  # type: ignore
            if country not in country_city_map:
                country_city_map[country] = []
            country_city_map[country].append(city)

        fig.update_yaxes(
            tickfont=dict(size=TEXT_SIZE, color="black"),
            showticklabels=True,  # Ensure city names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=left_margin,
                                      r=right_margin,
                                      t=top_margin,
                                      b=bottom_margin))
        self.save_plotly_figure(fig=fig,
                                filename=filename,
                                width=2400,
                                height=TALL_FIG_HEIGHT,
                                scale=SCALE,
                                save_eps=False,
                                save_final=True)

    def speed_and_time_to_start_cross_country(self, df_mapping, font_size_captions=40, x_axis_title_height=150,
                                              legend_x=0.81, legend_y=0.98, legend_spacing=0.02):
        logger.info("Plotting speed_and_time_to_start_cross")
        final_dict = {}
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        avg_speed = data_tuple[27]
        avg_time = data_tuple[28]

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
            country, condition = country_condition.split('_')

            # Get the iso3 from the mapping file
            iso_code = values_class.get_value(df=df_mapping,
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
                logger.info(f"{wrapper_class.format_city_state(country)}: {diff}")

            logger.info("Top 5 cities with min |speed_0 - speed_1| differences:")
            for country, diff in top_5_min_speed:
                logger.info(f"{wrapper_class.format_city_state(country)}: {diff}")
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
                logger.info(f"{wrapper_class.format_city_state(country)}: {diff}")

            logger.info("Top 5 cities with min |time_0 - time_1| differences:")
            for country, diff in top_5_min:
                logger.info(f"{wrapper_class.format_city_state(country)}: {diff}")
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

            logger.info(f"Country with max speed at day: {wrapper_class.format_city_state(max_speed_country_0)} with speed of {max_speed_value_0} m/s")  # noqa:E501
            logger.info(f"Country with min speed at day: {wrapper_class.format_city_state(min_speed_country_0)} with speed of {min_speed_value_0} m/s")  # noqa:E501

        if filtered_dict_s_1:
            max_speed_country_1 = max(filtered_dict_s_1, key=lambda country: filtered_dict_s_1[country]["speed_1"])
            min_speed_country_1 = min(filtered_dict_s_1, key=lambda country: filtered_dict_s_1[country]["speed_1"])
            max_speed_value_1 = filtered_dict_s_1[max_speed_country_1]["speed_1"]
            min_speed_value_1 = filtered_dict_s_1[min_speed_country_1]["speed_1"]

            logger.info(f"Country with max speed at night: {wrapper_class.format_city_state(max_speed_country_1)} with speed of {max_speed_value_1} m/s")  # noqa:E501
            logger.info(f"Country with min speed at night: {wrapper_class.format_city_state(min_speed_country_1)} with speed of {min_speed_value_1} m/s")  # noqa:E501

        # Find country with max and min time_0 and time_1
        if filtered_dict_t_0:
            max_time_country_0 = max(filtered_dict_t_0, key=lambda country: filtered_dict_t_0[country]["time_0"])
            min_time_country_0 = min(filtered_dict_t_0, key=lambda country: filtered_dict_t_0[country]["time_0"])
            max_time_value_0 = filtered_dict_t_0[max_time_country_0]["time_0"]
            min_time_value_0 = filtered_dict_t_0[min_time_country_0]["time_0"]

            logger.info(f"Country with max time at day: {wrapper_class.format_city_state(max_time_country_0)} with time of {max_time_value_0} s")  # noqa:E501
            logger.info(f"Country with min time at day: {wrapper_class.format_city_state(min_time_country_0)} with time of {min_time_value_0} s")  # noqa:E501

        if filtered_dict_t_1:
            max_time_country_1 = max(filtered_dict_t_1, key=lambda country: filtered_dict_t_1[country]["time_1"])
            min_time_country_1 = min(filtered_dict_t_1, key=lambda country: filtered_dict_t_1[country]["time_1"])
            max_time_value_1 = filtered_dict_t_1[max_time_country_1]["time_1"]
            min_time_value_1 = filtered_dict_t_1[min_time_country_1]["time_1"]

            logger.info(f"Country with max time at night: {wrapper_class.format_city_state(max_time_country_1)} with time of {max_time_value_1} s")  # noqa:E501
            logger.info(f"Country with min time at night: {wrapper_class.format_city_state(min_time_country_1)} with time of {min_time_value_1} s")  # noqa:E501

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
        TALL_FIG_HEIGHT = num_cities_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_cities_per_col * 2, cols=2,  # Two columns
            vertical_spacing=0,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[2.0] * (num_cities_per_col * 2),
        )

        # Plot left column (first half of cities)
        for i, country in enumerate(countries_ordered[:num_cities_per_col]):
            iso_code = values_class.get_value(df_mapping, "country", country, None, None, "iso3")
            # build up textual label for left column
            iso2 = wrapper_class.iso3_to_iso2(iso_code)
            # country = Analysis.iso2_to_flag(iso2) + " " + iso_code + " " + country
            country = wrapper_class.iso2_to_flag(iso2) + " " + country
            # Row for speed (Day and Night)
            row = 2 * i + 1
            if day_avg_speed[i] is not None and night_avg_speed[i] is not None:
                value = (day_avg_speed[i] + night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_avg_speed[i] is not None:  # Only day data available
                value = (day_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_avg_speed[i] is not None:  # Only night data available
                value = (night_avg_speed[i])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during night",
                    marker=dict(color=bar_colour_2), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            # Row for time (Day and Night)
            row = 2 * i + 2
            if day_time_dict[i] is not None and night_time_dict[i] is not None:
                value = (day_time_dict[i] + night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during night", marker=dict(color=bar_colour_4), text=[''],
                    textposition='auto', showlegend=False), row=row, col=1)

            elif day_time_dict[i] is not None:  # Only day time data available
                value = (day_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

            elif night_time_dict[i] is not None:  # Only night time data available
                value = (night_time_dict[i])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[i]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during night", marker=dict(color=bar_colour_4),
                    text=[''], textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=1)

        # Similarly for the right column
        for i, country in enumerate(countries_ordered[num_cities_per_col:]):
            iso_code = values_class.get_value(df_mapping, "country", country, None, None, "iso3")
            row = 2 * i + 1
            idx = num_cities_per_col + i
            # build up textual label for left column
            iso2 = wrapper_class.iso3_to_iso2(iso_code)
            # country = Analysis.iso2_to_flag(iso2) + " " + iso_code + " " + country
            country = wrapper_class.iso2_to_flag(iso2) + " " + country
            if day_avg_speed[idx] is not None and night_avg_speed[idx] is not None:
                value = (day_avg_speed[idx] + night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_avg_speed[idx] is not None:
                value = (day_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_avg_speed[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during day", marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_avg_speed[idx] is not None:
                value = (night_avg_speed[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_avg_speed[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} speed during night", marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            row = 2 * i + 2
            if day_time_dict[idx] is not None and night_time_dict[idx] is not None:
                value = (day_time_dict[idx] + night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during night", marker=dict(color=bar_colour_4), text=[''],
                    textposition='inside', showlegend=False), row=row, col=2)

            elif day_time_dict[idx] is not None:  # Only day time data available
                value = (day_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[day_time_dict[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during day", marker=dict(color=bar_colour_3),
                    text=[''], textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=28, color='white')), row=row, col=2)

            elif night_time_dict[idx] is not None:  # Only night time data available
                value = (night_time_dict[idx])/2
                fig.add_trace(go.Bar(
                    x=[night_time_dict[idx]], y=[f'{country} {value:.2f}'], orientation='h',
                    name=f"{country} time during night", marker=dict(color=bar_colour_4),
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
            {"name": "Mean speed of crossing during day (in m/s)", "color": bar_colour_1},
            {"name": "Mean speed of crossing during night (in m/s)", "color": bar_colour_2},
            {"name": "Mean time to start crossing during day (in s)", "color": bar_colour_3},
            {"name": "Mean time to start crossing during night (in s) ", "color": bar_colour_4},
        ]

        # Add the vertical legends at the top and bottom
        self.add_vertical_legend_annotations(fig, legend_items, x_position=legend_x, y_start=legend_y,
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
        self.save_plotly_figure(fig, "consolidated", height=TALL_FIG_HEIGHT*2, width=4960, scale=SCALE,
                                save_final=True, save_eps=False)
