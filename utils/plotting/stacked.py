import common
import math
from tqdm import tqdm
import itertools
import pickle
import numpy as np
from custom_logger import CustomLogger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.plotting.layout import Layout
from utils.plotting import constants as C
from utils.core.iso import ISO
from utils.core.metadata import MetaData
from utils.core.grouping import Grouping
from utils.plotting.io import IO

layout_class = Layout()
iso_class = ISO()
metadata_class = MetaData()
grouping_class = Grouping()
plots_io_class = IO()
logger = CustomLogger(__name__)  # use custom logger

# File to store the city coordinates
file_results = 'results.pickle'


class Stacked:
    def __init__(self) -> None:
        pass

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
                country = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
                iso_code = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "iso3")

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
                iso_code = metadata_class.get_value(df=df_mapping,
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
            city = grouping_class.process_city_string(city, df_mapping)

            if order_by == "average":
                iso_code = metadata_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
                city = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + city  # type: ignore

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
                    marker=dict(color=C.BAR_COLOR_1),
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
                    marker=dict(color=C.BAR_COLOR_2),
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
                    marker=dict(color=C.BAR_COLOR_1),
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
                    marker=dict(color=C.BAR_COLOR_2),
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
            city = grouping_class.process_city_string(city, df_mapping)

            if order_by == "average":
                iso_code = metadata_class.get_value(df_mapping, "city", city_new, "lat", float(lat), "iso3")
                city = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + city  # type: ignore

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
                    marker=dict(color=C.BAR_COLOR_1),
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
                    marker=dict(color=C.BAR_COLOR_2),
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
                    marker=dict(color=C.BAR_COLOR_1),
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
                    marker=dict(color=C.BAR_COLOR_2),
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
                {"name": "Day", "color": C.BAR_COLOR_1},
                {"name": "Night", "color": C.BAR_COLOR_2},
            ]

            # Add the vertical legends at the top and bottom
            layout_class.add_vertical_legend_annotations(fig,
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
            font_size = C.FLAG_SIZE  # Font size for visibility

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
                iso2 = iso_class.iso3_to_iso2(country)
                country = country + iso_class.iso2_to_flag(iso2)
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
                iso2 = iso_class.iso3_to_iso2(country)
                country = country + iso_class.iso2_to_flag(iso2)
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
            tickfont=dict(size=C.TEXT_SIZE, color="black"),
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
        plots_io_class.save_plotly_figure(fig=fig,
                                          filename=filename,
                                          width=2400,
                                          height=TALL_FIG_HEIGHT,
                                          scale=C.SCALE,
                                          save_eps=False,
                                          save_final=True)

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
            iso_code = metadata_class.get_value(df=df_mapping,
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
            iso_code = metadata_class.get_value(df_mapping, "country", country, None, None, "iso3")

            # build up textual label for left column
            country = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + country

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
                    marker=dict(color=C.BAR_COLOR_1),
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
                    marker=dict(color=C.BAR_COLOR_2),
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
                    marker=dict(color=C.BAR_COLOR_1),
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
                    marker=dict(color=C.BAR_COLOR_2),
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
            iso_code = metadata_class.get_value(df_mapping, "country", country, None, None, "iso3")

            # build up textual label for left column
            country = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + country

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
                    marker=dict(color=C.BAR_COLOR_1),
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
                    marker=dict(color=C.BAR_COLOR_2),
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
                    marker=dict(color=C.BAR_COLOR_1),
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
                    marker=dict(color=C.BAR_COLOR_2),
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
                {"name": "Day", "color": C.BAR_COLOR_1},
                {"name": "Night", "color": C.BAR_COLOR_2},
            ]

            # Add the vertical legends at the top and bottom
            layout_class.add_vertical_legend_annotations(fig,
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
            tickfont=dict(size=C.TEXT_SIZE, color="black"),
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

        plots_io_class.save_plotly_figure(fig=fig,
                                          filename=filename,
                                          width=width,
                                          height=height,
                                          scale=C.SCALE,
                                          save_eps=True,
                                          save_final=True)
