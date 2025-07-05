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
from tqdm import tqdm
import plotly.express as px


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

    def stack_plot(self, df_mapping, order_by, metric, data_view, title_text, filename, font_size_captions=40,
                   x_axis_title_height=110, legend_x=0.92, legend_y=0.015, legend_spacing=0.02, left_margin=10,
                   right_margin=10):
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
        metric_index_map = {
            "speed": 25,
            "time": 24
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

    def get_mapbox_map(self, df, hover_data=None, file_name="mapbox_map", save_final=True):
        """Generate world map with cities using mapbox.

        Args:
            df (dataframe): dataframe with mapping info.
            hover_data (list, optional): list of params to show on hover.
            file_name (str, optional): name of file
        """
        # Draw map
        fig = px.scatter_map(df,
                             lat="lat",
                             lon="lon",
                             hover_data=hover_data,
                             hover_name="city",
                             color=df["continent"],
                             zoom=1.3)  # type: ignore
        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  # Reduce margins
            # modebar_remove=["toImage"],  # remove modebar image button
            # showlegend=False,  # hide legend if not needed
            # annotations=[],  # remove any extra annotations
            mapbox=dict(zoom=1.3),
            font=dict(family=common.get_configs('font_family'),  # update font family
                      size=common.get_configs('font_size'))  # update font size
        )
        # Save and display the figure
        self.save_plotly_figure(fig, file_name, save_final=True)

    def hist(self, data_index, name, nbins=None, color=None, pretty_text=False, marginal='rug',
             xaxis_title=None, yaxis_title=None, name_file=None, save_file=False, save_final=False,
             fig_save_width=1320, fig_save_height=680, font_family=None, font_size=None,
             vlines=None, xrange=None):
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
        with open('no_filter.txt', 'w') as f:
            for value in all_values:
                f.write(f"{value}\n")

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
