# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
import os
from utils.values import Values
from utils.wrappers import Wrappers
from utils.tools import Tools
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
tools_class = Tools()

# File to store the city coordinates
file_results = 'results.pickle'

# Colours in graphs
bar_colour_1 = 'rgb(251, 180, 174)'
bar_colour_2 = 'rgb(179, 205, 227)'
bar_colour_3 = 'rgb(204, 235, 197)'
bar_colour_4 = 'rgb(222, 203, 228)'

# Consts
BASE_HEIGHT_PER_ROW = 30  # Adjust as needed
FLAG_SIZE = 22
TEXT_SIZE = 22
SCALE = 1  # scale=3 hangs often


class Plots():
    def __init__(self) -> None:
        pass

    def add_vertical_legend_annotations(self, fig, legend_items, x_position, y_start, spacing=0.03, font_size=50):
        """Adds vertical legend annotations to a Plotly figure.

        This method creates a vertical list of legend items in a Plotly figure,
        positioning them at a specified location and spacing. Each legend item
        consists of a colored square followed by its name.

        Args:
            fig (plotly.graph_objs.Figure): The Plotly figure to which annotations will be added.
            legend_items (list[dict]): A list of dictionaries where each dictionary
                contains:
                    - "color" (str): The hex or named color of the legend symbol.
                    - "name" (str): The display name for the legend item.
            x_position (float): The horizontal position of the legend in paper coordinates (0 to 1).
            y_start (float): The vertical starting position of the first legend item in paper coordinates.
            spacing (float, optional): The vertical space between legend items. Defaults to 0.03.
            font_size (int, optional): The font size for legend text. Defaults to 50.

        Returns:
            None: The figure is modified in-place.
        """
        # Loop through each legend item and add an annotation
        for i, item in enumerate(legend_items):
            fig.add_annotation(
                x=x_position,  # X position of the annotation (relative to paper coordinates)
                y=y_start - i * spacing,  # Adjust Y position for each item using spacing
                xref='paper',  # X coordinate is relative to the entire figure (paper)
                yref='paper',  # Y coordinate is relative to the entire figure (paper)
                showarrow=False,  # No arrow for the annotation
                text=f'<span style="color:{item["color"]};">&#9632;</span> {item["name"]}',  # Colored square + name
                font=dict(size=font_size),  # Font size for the annotation text
                xanchor='left',  # Align text starting from the left
                align='left'  # Text alignment within the annotation
            )

    def save_plotly_figure(self, fig, filename, width=1600, height=900, scale=SCALE, save_final=True, save_png=True,
                           save_eps=True):
        """
        Saves a Plotly figure as HTML, PNG, SVG, and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): width of the PNG and EPS images in pixels. Defaults to 1600.
            height (int, optional): height of the PNG and EPS images in pixels. Defaults to 900.
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
        except ValueError as e:
            logger.error(f"Value error raised when attempted to save image {filename}: {e}")

    def stack_plot(self, df_mapping, order_by, metric, data_view, title_text, filename, analysis_level="city",
                   font_size_captions=40, x_axis_title_height=110, legend_x=0.92, legend_y=0.015, legend_spacing=0.02,
                   left_margin=10, right_margin=10):
        """
        Plots a stacked bar graph based on the provided data and configuration.

        Parameters:
            df_mapping (dict): A dictionary mapping categories to their respective DataFrames.
            order_by (str): Criterion to order the bars, e.g., 'alphabetical' or 'average'.
            metric (str): The metric to visualise, such as 'speed' or 'time'.
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
                    value = (day_values[idx] + night_values[idx])/2
                else:
                    value = (day_values[idx] + night_values[idx])

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
            start, step, count = 1, 1, 26

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
        """Generates a world map of cities using Mapbox, with optional density visualization.

        This method can create either:
            1. A simple scatter map showing city locations colored by continent.
            2. A density map showing intensity values based on a specified column.

        Args:
            df (pandas.DataFrame): DataFrame containing mapping information.
                Required columns: "lat", "lon", "city", "continent".
            density_col (str, optional): Column name for density values.
                If provided, a density map is generated. Defaults to None.
            density_radius (int, optional): The pixel radius for density spread. Defaults to 30.
            hover_data (list, optional): List of additional DataFrame columns to display when hovering.
                Defaults to None.
            file_name (str, optional): Name of the saved file (without extension). Defaults to "mapbox_map".
            save_final (bool, optional): If True, saves the figure. Defaults to True.

        Returns:
            None: The Plotly figure is created, displayed, and optionally saved.
        """
        # Draw scatter map if no density column is provided
        if not density_col:
            fig = px.scatter_map(
                df,
                lat="lat",
                lon="lon",
                hover_data=hover_data,
                hover_name="city",
                color=df["continent"],  # Color points by continent
                zoom=1.3  # Initial zoom level # pyright: ignore[reportArgumentType]
            )
        # Draw density map if density column is provided
        else:
            fig = px.density_mapbox(
                df,
                lat="lat",
                lon="lon",
                z=density_col,  # Use density column for intensity
                radius=density_radius,  # Control the spread of density
                zoom=2.5,  # Initial zoom level for density view # pyright: ignore[reportArgumentType]
                center=dict(
                    lat=df["lat"].mean(),
                    lon=df["lon"].mean()
                ),  # Center map on mean coordinates
                mapbox_style="carto-positron",  # Light and clean map style
                hover_name="city",
                hover_data=hover_data,
            )

        # Update map layout to improve appearance
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
            mapbox=dict(zoom=1.3),
            font=dict(
                family=common.get_configs('font_family'),
                size=common.get_configs('font_size')
            ),
            legend=dict(
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.6)',  # Semi-transparent white background
                bordercolor='rgba(0,0,0,0.1)',   # Light border
                borderwidth=1
            ),
            legend_title_text=""  # Remove legend title
        )

        # Save the figure if requested
        self.save_plotly_figure(fig, file_name, save_final=save_final)

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

    def hist(self, data_index, name, min_threshold, max_threshold, nbins=None, raw=True, color=None,
             pretty_text=False, marginal='rug', xaxis_title=None, yaxis_title=None,
             name_file=None, df_mapping=None, save_file=False, save_final=False, fig_save_width=1320,
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

        if name == "speed" and raw is True:
            all_values = [speed for city in nested_dict.values() for video in city.values() for speed in video.values()]  # noqa:E501
            all_values = [value for value in all_values
                          if (min_threshold is None or value >= min_threshold)
                          and (max_threshold is None or value <= max_threshold)]

        elif name == "time" and raw is True:
            all_values = []
            for key, values in nested_dict.items():
                all_values.extend(values)

        elif name == "speed_filtered" and raw is False:
            no_of_crossing = data_tuple[35]
            all_values = []
            for key, values in nested_dict.items():
                if no_of_crossing[key] >= common.get_configs("min_crossing_detect"):
                    all_values.extend(values)

        elif name == "time_filtered" and raw is False:
            no_of_crossing = data_tuple[35]
            all_values = []
            for key, values in nested_dict.items():
                city, lat, long, cond = key.split("_")
                country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
                if country is not None:
                    if no_of_crossing[f"{country}_{cond}"] >= common.get_configs("min_crossing_detect"):
                        all_values.extend(values)

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
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            self.save_plotly_figure(fig, name_file, save_final=True, width=fig_save_width, height=fig_save_height)
        else:
            fig.show()

    def violin_plot(self, data_index, name, min_threshold, max_threshold, df_mapping, color=None,
                    pretty_text=False, xaxis_title=None, yaxis_title=None,
                    name_file=None, save_file=False, save_final=False, fig_save_width=1320,
                    fig_save_height=680, font_family=None, font_size=None, vlines=None, xrange=None):
        """Generates a violin plot grouped by country from nested results data.

        This method loads results from a pickle file, extracts values per country,
        and creates violin plots for visualizing distributions.
        It groups the data by country (based on a city-lat mapping) and displays
        box-and-mean lines within each violin.

        Args:
            data_index (int): Index of the dataset within the loaded pickle's tuple.
            name (str): Name associated with the dataset or plot.
            min_threshold (float): Minimum value filter for data (currently unused).
            max_threshold (float): Maximum value filter for data (currently unused).
            df_mapping (pandas.DataFrame): DataFrame for mapping city/lat to country.
            color (str, optional): Color for violin plots. Defaults to None.
            pretty_text (bool, optional): If True, formats axis labels nicely. Defaults to False.
            xaxis_title (str, optional): Custom title for the x-axis. Defaults to "Country".
            yaxis_title (str, optional): Custom title for the y-axis. Defaults to "Values".
            name_file (str, optional): Name for saving the figure file. Defaults to None.
            save_file (bool, optional): If True, saves the plot as an intermediate file. Defaults to False.
            save_final (bool, optional): If True, saves the final plot. Defaults to False.
            fig_save_width (int, optional): Width of the saved figure in pixels. Defaults to 1320.
            fig_save_height (int, optional): Height of the saved figure in pixels. Defaults to 680.
            font_family (str, optional): Font family for text in the figure. Defaults to None.
            font_size (int, optional): Font size for text in the figure. Defaults to None.
            vlines (list, optional): List of vertical lines to add to the plot. Defaults to None.
            xrange (tuple, optional): X-axis range (min, max). Defaults to None.

        Returns:
            None: The Plotly figure is displayed (and optionally saved).
        """
        # Load data from pickle file
        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        # Extract the nested dictionary at the given index
        nested_dict = data_tuple[data_index]

        values = {}

        # Loop through each (city, lat, ...) key and collect values grouped by country
        for city_lat_long_cond, inner_dict in nested_dict.items():
            city, lat, _, _ = city_lat_long_cond.split("_")
            # Get country from mapping DataFrame
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            if country not in values:
                values[country] = []  # Initialize a list for each country
            # Collect all values from the nested structure
            for video_id, inner_values in inner_dict.items():
                values[country].extend(inner_values.values())

        # Create the figure
        fig = go.Figure()

        # Add one violin plot per country
        for country, country_values in values.items():
            fig.add_trace(go.Violin(
                y=country_values,
                x=[country] * len(country_values),  # Repeat country name for each value
                name=country,
                box_visible=True,         # Show inner box plot
                meanline_visible=True,    # Show mean line
                # points='all'            # Optional: show individual points
            ))

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title=xaxis_title or "Country",
            yaxis_title=yaxis_title or "Values",
            violinmode='group'  # Group violins side-by-side if multiple categories
        )

        # Display the plot
        fig.show()

    def map_world(self, df, *, color, title=None, projection="natural earth", hover_name="country", hover_data=None,
                  show_colorbar=False, colorbar_title=None, colorbar_kwargs=None, color_scale="YlOrRd",
                  show_cities=False, df_cities=None, city_marker_size=3, show_images=False, image_items=None,
                  denmark_greenland=False, save_file=False, save_final=False, file_basename="map",
                  filter_zero_nan=True, country_name_map=None):
        """
        Unified world choropleth with optional cities and annotations.

        Args:
            df (pd.DataFrame): Must include 'country' and the column referenced by `color`.
            color (str): Column in df for choropleth coloring.
            title (str|None): Optional figure title.
            projection (str): Plotly geo projection (e.g., 'natural earth').
            hover_name (str): Column used for hover name (defaults to 'country').
            hover_data (list|dict|None): Extra hover fields.
            show_colorbar (bool): Whether to show a color bar.
            colorbar_title (str|None): Title for color bar (defaults to `color`).
            colorbar_kwargs (dict|None): Extra layout props for the color bar (merged with sensible defaults).
            color_scale (str|list): Plotly color scale.
            show_cities (bool): Add city markers from `df_cities`.
            df_cities (pd.DataFrame|None): Needs columns 'lat' and 'lon'; optional 'city','country'.
            city_marker_size (int|float): Marker size for cities.
            show_images (bool): Add overlay images/arrows/labels from `image_items`.
            image_items (list[dict]|None): Same schema you used before.
            denmark_greenland (bool): If True, duplicate Denmark’s value for Greenland.
            save_file (bool): If True, save HTML (and optionally final image) via your helper.
            save_final (bool): Passed to your `save_plotly_figure`.
            file_basename (str): Base filename without extension.
            filter_zero_nan (bool): Filter rows where `color` is 0 or NaN (your old map() behavior).
            country_name_map (dict|None): Extra name normalization; default includes {'Türkiye': 'Turkey'}.
        """

        # --- prep ---
        df = df.copy()
        if country_name_map is None:
            country_name_map = {"Türkiye": "Turkey"}
        df["country"] = df["country"].replace(country_name_map)

        # Optional Greenland-from-Denmark duplication
        if denmark_greenland and "country" in df and color in df:
            denmark_vals = df.loc[df["country"] == "Denmark", color].values
            if len(denmark_vals) > 0:
                # build row with same columns
                greenland_row = {col: (None if col not in ("country", color) else
                                       ("Greenland" if col == "country" else denmark_vals[0]))
                                 for col in df.columns}
                df = pd.concat([df, pd.DataFrame([greenland_row])], ignore_index=True)

        # Optional filter like old map()
        if filter_zero_nan:
            df = df[df[color].fillna(0) != 0].copy()

        # Default colorbar title
        if colorbar_title is None:
            colorbar_title = color

        # --- figure ---
        fig = px.choropleth(
            df,
            locations="country",
            locationmode="country names",
            color=color,
            hover_name=hover_name,
            hover_data=hover_data,
            projection=projection,
            color_continuous_scale=color_scale,
            title=title,
        )

        # Fonts from your config
        fig.update_layout(
            font=dict(
                family=common.get_configs('font_family'),
                size=common.get_configs('font_size'),
            )
        )

        # --- optional cities ---
        if show_cities and df_cities is not None and {"lat", "lon"}.issubset(df_cities.columns):
            fig.add_trace(go.Scattergeo(
                lon=df_cities["lon"],
                lat=df_cities["lat"],
                text=df_cities.get("city", None),
                mode="markers",
                hoverinfo="skip",
                marker=dict(size=city_marker_size, color="black", opacity=0.7, symbol="circle"),
                name="cities",
            ))

        # --- optional image overlays/arrows/labels (your schema preserved) ---
        if show_images and image_items:
            # base path reused from your code
            path_screenshots = os.path.join(common.root_dir, "readme")

            # images and labels
            for item in image_items:
                if "file" in item:
                    fig.add_layout_image(dict(
                        source=os.path.join(path_screenshots, item["file"]),
                        xref="paper", yref="paper",
                        x=item.get("x", 0.5), y=item.get("y", 0.5),
                        sizex=item.get("sizex", 0.1), sizey=item.get("sizey", 0.1),
                        xanchor=item.get("xanchor", "center"),
                        yanchor=item.get("yanchor", "middle"),
                        layer=item.get("layer", "above"),
                    ))
                if "label" in item:
                    fig.add_annotation(
                        text=item["label"],
                        x=item.get("x_label", item.get("x", 0.5)),
                        y=item.get("y_label", item.get("y", 0.5) + 0.1),
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black",
                        borderwidth=1,
                    )

            # lines to actual city coords (if df_cities available)
            if df_cities is not None and {"lat", "lon", "city", "country"}.issubset(df_cities.columns):
                for item in image_items:
                    if "city" in item and "country" in item and \
                       "approx_lon" in item and "approx_lat" in item:
                        row = df_cities[
                            (df_cities["city"].str.lower() == item["city"].lower()) &
                            (df_cities["country"].str.lower() == item["country"].lower())
                        ]
                        if not row.empty:
                            fig.add_trace(go.Scattergeo(
                                lon=[item["approx_lon"], row["lon"].values[0]],
                                lat=[item["approx_lat"], row["lat"].values[0]],
                                mode="lines",
                                line=dict(width=2, color="black"),
                                showlegend=False,
                                geo="geo",
                                hoverinfo="skip"
                            ))
                    # optional small “video code” tag
                    if "video" in item:
                        fig.add_annotation(dict(
                            text=item["video"],
                            x=item.get("x_video", item.get("x", 0.5)),
                            y=item.get("y_video", item.get("y", 0.5) - 0.1),
                            xref="paper", yref="paper",
                            showarrow=False,
                            font=dict(size=10, color="black"),
                            align="center",
                            bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="black",
                            borderwidth=1,
                        ))

        # --- colorbar handling (merged defaults + overrides) ---
        base_colorbar = dict(
            title=colorbar_title,
            orientation="h",
            x=0.5,
            y=0.06,
            xanchor="center",
            yanchor="bottom",
            len=0.5,
            thickness=10,
            bgcolor="rgba(255,255,255,0.7)",
            tickfont=dict(size=max(common.get_configs('font_size') - 5, 8)),
        )
        if colorbar_kwargs:
            # shallow merge, keys in colorbar_kwargs win
            base_colorbar.update(colorbar_kwargs)

        fig.update_coloraxes(
            showscale=show_colorbar,
            colorbar=(base_colorbar if show_colorbar else {})
        )

        # --- layout ---
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )

        # --- save/show ---
        if save_file:
            # tiny padding for saved version
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            # save via your utility
            self.save_plotly_figure(fig, file_basename, save_final=save_final)
        else:
            fig.show()

        return fig

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
        if len(df) > 0:
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
            df = df[np.isfinite(df[[x, y]]).all(axis=1)].copy()  # Remove NaNs and Infs

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
                           legend_x=0.87, legend_y=0.04, font_size_captions=40, raw=False, legend_spacing=0.02,
                           left_margin=10, right_margin=10, top_margin=0, bottom_margin=0, height=2400, width=2480):
        """
        Plots a stacked bar graph based on the provided data and configuration.

        Parameters:
            df_mapping (dict): A dictionary mapping categories to their respective DataFrames.
            order_by (str): Criterion to order the bars, e.g., 'alphabetical' or 'average'.
            metric (str): The metric to visualise, such as 'speed' or 'time'.
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

        no_of_crossing = data_tuple[35]

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
            if not raw:
                if no_of_crossing[country_condition] < common.get_configs("min_crossing_detect"):
                    continue
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
            all_sd = [final_dict[country][all_sd_key] for country in countries_ordered]

        elif data_view == "night":
            day_values = [0 for country in countries_ordered]
            day_sd = [0 for country in countries_ordered]
            night_values = [final_dict[country][night_key] for country in countries_ordered]
            night_sd = [final_dict[country][night_sd_key] for country in countries_ordered]
            all_sd = [final_dict[country][all_sd_key] for country in countries_ordered]

        # Determine how many countries will be in each column
        num_countries_per_col = len(countries_ordered) // 2 + len(countries_ordered) % 2  # Split cities

        # Define a base height per row and calculate total figure height
        # TALL_FIG_HEIGHT = num_countries_per_col * BASE_HEIGHT_PER_ROW

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

                y_value = [
                    f"{country} {value:.2f}±{all_sd[i]:.2f} "
                    f"(D={day_values[i]:.2f}±{day_sd[i]:.2f}, "
                    f"N={night_values[i]:.2f}±{night_sd[i]:.2f})"]

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
                y_value = [f"{country} {value:.2f}±{day_sd[i]:.2f}"]

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
                y_value = [f"{country} {value:.2f}±{night_sd[i]:.2f}"]

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
                y_value = [
                    f"{country} {value:.2f}±{all_sd[idx]:.2f} "
                    f"(D={day_values[idx]:.2f}±{day_sd[idx]:.2f}, "
                    f"N={night_values[idx]:.2f}±{night_sd[idx]:.2f})"]

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
                y_value = [f"{country} {value:.2f}±{day_sd[idx]:.2f}"]

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
                y_value = [f"{country} {value:.2f}±{night_sd[idx]:.2f}"]

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
            height=height,
            width=width,
            showlegend=False,  # Hide the default legend
            margin=dict(t=0, b=0),
            bargap=0,
            bargroupgap=0
        )

        # Define gridline generation parameters
        if metric == "speed":
            start, step, count = 0.5, 0.5, 9
        elif metric == "time":
            start, step, count = 2, 2, 30

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
        # Speed (main text)
        # fig.add_annotation(
        #     text="0.5",
        #     xref="paper", yref="paper",
        #     x=0.06, y=1.032,  # adjust these values to position the label above the plot
        #     showarrow=False,
        #     font=dict(
        #         size=common.get_configs("font_size")+28,
        #         family=common.get_configs("font_family")
        #     )
        # )

        # fig.add_annotation(
        #     text="1",
        #     xref="paper", yref="paper",
        #     x=0.142, y=1.032,  # adjust these values to position the label above the plot
        #     showarrow=False,
        #     font=dict(
        #         size=common.get_configs("font_size")+28,
        #         family=common.get_configs("font_family")
        #     )
        # )

        # Time (main text)
        fig.add_annotation(
            text="2",
            xref="paper", yref="paper",
            x=0.595, y=1.032,  # adjust these values to position the label above the plot
            showarrow=False,
            font=dict(
                size=common.get_configs("font_size")+28,
                family=common.get_configs("font_family")
            )
        )

        # # Time (appendix)
        # fig.add_annotation(
        #     text="1",
        #     xref="paper", yref="paper",
        #     x=0.576, y=1.07,  # adjust these values to position the label above the plot
        #     showarrow=False,
        #     font=dict(
        #         size=common.get_configs("font_size")+28,
        #         family=common.get_configs("font_family")
        #     )
        # )

        self.save_plotly_figure(fig=fig,
                                filename=filename,
                                width=width,
                                height=height,
                                scale=SCALE,
                                save_eps=True,
                                save_final=True)

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
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            continent = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "continent")
            population_country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "population_country")  # noqa: E501
            gdp_city = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "gmp")
            traffic_mortality = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "traffic_mortality")  # noqa: E501
            literacy_rate = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "literacy_rate")
            gini = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "gini")
            traffic_index = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "traffic_index")

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

        self.save_plotly_figure(fig, "correlation_matrix_heatmap_day", save_final=True)

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

        self.save_plotly_figure(fig, "correlation_matrix_heatmap_night", save_final=True)

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

        self.save_plotly_figure(fig, "correlation_matrix_heatmap_averaged", save_final=True)

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

            self.save_plotly_figure(fig, f"correlation_matrix_heatmap_{continents}", save_final=True)

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
        ped_cross_city = wrapper_class.country_sum_from_cities(ped_cross_city, df_mapping)
        person_city = wrapper_class.country_averages_from_nested(person_city, df_mapping)
        bicycle_city = wrapper_class.country_averages_from_nested(bicycle_city, df_mapping)
        car_city = wrapper_class.country_averages_from_nested(car_city, df_mapping)
        motorcycle_city = wrapper_class.country_averages_from_nested(motorcycle_city, df_mapping)
        bus_city = wrapper_class.country_averages_from_nested(bus_city, df_mapping)
        truck_city = wrapper_class.country_averages_from_nested(truck_city, df_mapping)
        vehicle_city = wrapper_class.country_averages_from_nested(vehicle_city, df_mapping)
        cellphone_city = wrapper_class.country_averages_from_nested(cellphone_city, df_mapping)
        trf_sign_city = wrapper_class.country_averages_from_nested(trf_sign_city, df_mapping)
        cross_evnt_city = wrapper_class.country_averages_from_flat(cross_evnt_city, df_mapping)

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
            iso_code = values_class.get_value(df_mapping, "country", country, None, None, "iso3")
            continent = values_class.get_value(df_mapping, "country", country, None, None, "continent")
            traffic_mortality = values_class.get_value(df_mapping, "country", country, None, None, "traffic_mortality")
            literacy_rate = values_class.get_value(df_mapping, "country", country, None, None, "literacy_rate")
            gini = values_class.get_value(df_mapping, "country", country, None, None, "gini")
            med_age = values_class.get_value(df_mapping, "country", country, None, None, "med_age")
            avg_day_night_speed = values_class.get_value(df_countries, "country", country,
                                                         None, None, "speed_crossing_day_night_country_avg")
            avg_day_night_time = values_class.get_value(df_countries, "country", country,
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

        self.save_plotly_figure(fig, "correlation_matrix_heatmap_day", save_final=True)

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

        self.save_plotly_figure(fig, "correlation_matrix_heatmap_night", save_final=True)

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
        corr_matrix_avg = agg_df.corr(method='spearman')

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

        self.save_plotly_figure(fig, "correlation_matrix_heatmap_averaged", save_final=True)

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
            corr_matrix_avg = agg_df.corr(method='spearman')

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
                self.save_plotly_figure(fig, f"correlation_matrix_heatmap_{continents}", save_final=True)
            # open it in localhost instead
            else:
                fig.show()

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
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:
                # Initialise the city's dictionary if not already present
                if f"{city}_{lat}_{long}" not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {"without_trf_light_0": None, "without_trf_light_1": None,
                                                          "country": country, "iso": iso_code}

                # normalise by total time and total number of detected persons
                total_time = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "total_time")
                person = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "person")
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
            key=lambda city: values_class.safe_average([
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
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            if day_crossing[i] is not None and night_crossing[i] is not None:
                value = round((day_crossing[i] + night_crossing[i])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=1)

            elif day_crossing[i] is not None:  # Only day data available
                value = (day_crossing[i])
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_crossing[i] is not None:  # Only night data available
                value = (night_crossing[i])
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            idx = num_cities_per_col + i
            if day_crossing[idx] is not None and night_crossing[idx] is not None:
                value = round((day_crossing[idx] + night_crossing[idx])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_crossing[idx] is not None:
                value = (day_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_crossing[idx] is not None:
                value = (night_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing without traffic light in night",
                    marker=dict(color=bar_colour_2),
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
            {"name": "Day", "color": bar_colour_1},
            {"name": "Night", "color": bar_colour_2},
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
        self.save_plotly_figure(fig,
                                "crossings_without_traffic_equipment_avg",
                                width=2480,
                                height=TALL_FIG_HEIGHT,
                                scale=SCALE,
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
            country = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
            iso_code = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")
            if country or iso_code is not None:
                # Initialise the city's dictionary if not already present
                if f"{city}_{lat}_{long}" not in final_dict:
                    final_dict[f"{city}_{lat}_{long}"] = {"with_trf_light_0": None, "with_trf_light_1": None,
                                                          "country": country, "iso": iso_code}

                # normalise by total time and total number of detected persons
                total_time = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "total_time")
                person = values_class.get_value(df_mapping, "city", city, "lat", float(lat), "person")
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
            key=lambda city: values_class.safe_average([
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
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            if day_crossing[i] is not None and night_crossing[i] is not None:
                value = round((day_crossing[i] + night_crossing[i])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='auto', showlegend=False), row=row, col=1)

            elif day_crossing[i] is not None:  # Only day data available
                value = (day_crossing[i])
                fig.add_trace(go.Bar(
                    x=[day_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=1)

            elif night_crossing[i] is not None:  # Only night data available
                value = (night_crossing[i])
                fig.add_trace(go.Bar(
                    x=[night_crossing[i]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    textposition='auto', insidetextanchor='start', showlegend=False,
                    text=[''], textfont=dict(size=14, color='white')), row=row, col=1)

        for i, city in enumerate(cities_ordered[num_cities_per_col:]):
            city_new, lat, long = city.split('_')
            iso_code = values_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
            city = wrapper_class.process_city_string(city, df_mapping)

            city = wrapper_class.iso2_to_flag(wrapper_class.iso3_to_iso2(iso_code)) + " " + city   # type: ignore

            row = i + 1
            idx = num_cities_per_col + i
            if day_crossing[idx] is not None and night_crossing[idx] is not None:
                value = round((day_crossing[idx] + night_crossing[idx])/2, 2)
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
                    text=[''], textposition='inside', showlegend=False), row=row, col=2)

            elif day_crossing[idx] is not None:
                value = (day_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[day_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in day",
                    marker=dict(color=bar_colour_1), text=[''],
                    textposition='inside', insidetextanchor='start', showlegend=False,
                    textfont=dict(size=14, color='white')), row=row, col=2)

            elif night_crossing[idx] is not None:
                value = (night_crossing[idx])
                fig.add_trace(go.Bar(
                    x=[night_crossing[idx]], y=[f'{city} {value}'], orientation='h',
                    name=f"{city} crossing with traffic light in night",
                    marker=dict(color=bar_colour_2),
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
            {"name": "Day", "color": bar_colour_1},
            {"name": "Night", "color": bar_colour_2},
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
        self.save_plotly_figure(fig,
                                "crossings_with_traffic_equipment_avg",
                                width=2480,
                                height=TALL_FIG_HEIGHT,
                                scale=SCALE,
                                save_eps=False,
                                save_final=True)
