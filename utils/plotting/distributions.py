import common
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from custom_logger import CustomLogger
from utils.core.metadata import MetaData
from utils.plotting.io import IO

logger = CustomLogger(__name__)  # use custom logger
metadata_class = MetaData()
io_class = IO()


class Distributions:
    def __init__(self) -> None:
        pass

    def hist(self, data_index, name, min_threshold, max_threshold, nbins=None, raw=True, color=None,
             pretty_text=False, marginal='rug', xaxis_title=None, yaxis_title=None,
             file_name=None, df_mapping=None, save_file=False, save_final=False, fig_save_width=1320,
             fig_save_height=680, font_family=None, font_size=None, vlines=None, xrange=None, data_file=None):
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
            file_name (str, optional): name of file to save.
            save_file (bool, optional): whether to save HTML file of the plot.
            save_final (bool, optional): whether to save final figure to /figures.
            fig_save_width (int, optional): width of saved figure.
            fig_save_height (int, optional): height of saved figure.
            font_family (str, optional): font family to use. Defaults to config.
            font_size (int, optional): font size to use. Defaults to config.
        """

        # Load data from pickle file
        with open(data_file, 'rb') as file:  # type: ignore
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
                country = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
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
            if not file_name:
                file_name = f"hist_{name}"
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            io_class.save_plotly_figure(fig, file_name, save_final=True, width=fig_save_width, height=fig_save_height)
        else:
            fig.show()

    def video_histogram_by_month(self, df, video_count_col='video_count', upload_date_col='upload_date',
                                 xaxis_title='Upload Month', yaxis_title='Number of Videos',
                                 file_name=None, save_file=False, font_family=None, font_size=None,
                                 title=None):
        """
        Output a histogram of video counts by the month of upload_date, where upload_date is a string list of dates in
        DDMMYYYY format, potentially including None or 7-digit dates. Each month is a separate bar, with years shown on
        the x-axis.

        Args:
            df (pandas.DataFrame): DataFrame containing video_count and upload_date columns.
            video_count_col (str): Column with video counts (default: 'video_count').
            upload_date_col (str): Column with upload dates as string lists in DDMMYYYY format
                                   (default: 'upload_date').
            xaxis_title (str): Title for x-axis (default: 'Upload Month').
            yaxis_title (str): Title for y-axis (default: 'Number of Videos').
            file_name (str): Name of file to save (default: None, generates 'video_histogram_by_month').
            save_file (bool): Flag to save the plot as HTML (default: False).
            font_family (str): Font family for the figure (default: None, uses 'Arial').
            font_size (int): Font size for the figure (default: None, uses 12).
            title (str): Title of the histogram (default: 'Video Count by Upload Month').

        Returns:
            plotly.graph_objects.Figure: The generated histogram figure, or -1 if an error occurs.
        """
        if len(df) == 0:
            logger.error('DataFrame is empty.')
            return -1

        # Verify required columns
        required_cols = [video_count_col, upload_date_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f'Missing columns: {missing_cols}')
            return -1

        # Copy DataFrame to avoid modifying original
        df = df.copy()

        # Log data summary
        logger.info(f'Creating histogram for {video_count_col} by month of {upload_date_col}. Rows: {len(df)}')

        # Clean data: Remove rows with invalid video_count or upload_date
        df = df[df[video_count_col].notna() & (df[video_count_col] >= 0)]
        df = df[df[upload_date_col].notna() & (df[upload_date_col] != '')]
        if len(df) == 0:
            logger.error('No valid data after cleaning.')
            return -1

        # Parse and clean upload_date lists
        def parse_date_list(date_str):
            try:
                # Convert string representation of list to actual list, handling string input
                if isinstance(date_str, str):
                    # Remove square brackets and split by comma
                    dates = [d.strip() for d in date_str.strip('[]').split(',') if d.strip()]
                else:
                    dates = date_str
                valid_dates = []
                invalid_dates = []
                for d in dates:
                    if d is None or d == 'None':
                        invalid_dates.append('None')
                        continue
                    d_str = str(d).strip("'").strip('"')  # Remove any quotes
                    # Fix 7-digit dates by adding leading zero
                    if len(d_str) == 7 and d_str.isdigit():
                        d_str = '0' + d_str
                    # Validate: 8 digits, numeric, and valid DDMMYYYY
                    if len(d_str) == 8 and d_str.isdigit():
                        try:
                            day, month, year = int(d_str[:2]), int(d_str[2:4]), int(d_str[4:])
                            if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                                valid_dates.append(d_str)
                            else:
                                invalid_dates.append(d_str)
                        except ValueError:
                            invalid_dates.append(d_str)
                    else:
                        invalid_dates.append(d_str)
                return valid_dates, invalid_dates
            except Exception as e:
                logger.error(f'Error parsing date list {date_str}: {str(e)}')
                return [], [str(date_str)]

        # Apply parsing and collect invalid dates
        df['parsed_dates'], df['invalid_dates'] = zip(*df[upload_date_col].apply(parse_date_list))

        # Log invalid dates
        invalid_rows = df[df['invalid_dates'].apply(len) > 0]
        if not invalid_rows.empty:
            logger.warning(f'Invalid dates found in {len(invalid_rows)} rows. Sample invalid dates:')
            for idx, row in invalid_rows.head().iterrows():
                logger.warning(f'Row {idx}, city {row.get("city", "unknown")}: Invalid dates {row["invalid_dates"]}')

        # Verify video_count matches length of parsed_dates
        df['date_count'] = df['parsed_dates'].apply(len)
        mismatches = df[df['video_count'] != df['date_count']]
        if not mismatches.empty:
            logger.warning(f'Mismatch between video_count and parsed_dates length in {len(mismatches)} rows.')

        # Filter rows with at least one valid date
        df = df[df['parsed_dates'].apply(len) > 0]
        if len(df) == 0:
            logger.error('No rows with valid dates.')
            return -1

        # Explode parsed_dates to create one row per date
        df_exploded = df.explode('parsed_dates')
        df_exploded = df_exploded[df_exploded['parsed_dates'].notna() & (df_exploded['parsed_dates'] != '')]

        if len(df_exploded) == 0:
            logger.error('No valid dates after exploding.')
            return -1

        # Parse DDMMYYYY dates to year-month
        df_exploded['datetime'] = pd.to_datetime(df_exploded['parsed_dates'], format='%d%m%Y', errors='coerce')
        df_exploded['year_month'] = df_exploded['datetime'].dt.to_period('M').astype(str)
        df_exploded['year'] = df_exploded['datetime'].dt.year
        # Check for invalid year values
        invalid_years = df_exploded['year'].isna() | (df_exploded['year_month'] == 'NaT')
        if invalid_years.any():
            logger.warning(f'Invalid years or dates after parsing in {invalid_years.sum()} rows: ' +
                           f'{str(df_exploded[invalid_years][["parsed_dates", "year_month", "year"]].head().to_dict("records"))}')  # noqa: E501
            df_exploded = df_exploded[~invalid_years]

        if len(df_exploded) == 0:
            logger.error('No valid dates after parsing.')
            return -1

        if 'year' not in df_exploded.columns or df_exploded['year'].isna().all():
            logger.error('Year column is missing or contains only NaN values.')
            return -1
        video_counts = df_exploded.groupby(['year', 'year_month']).size().reset_index(name='video_count')
        video_counts['year_month'] = video_counts['year_month'].astype(str)

        # Create histogram
        try:
            fig = px.histogram(
                video_counts,
                x='year_month',
                y='video_count',
                title=title,
                labels={'year_month': xaxis_title, 'video_count': yaxis_title},
                nbins=len(video_counts['year_month'].unique())
            )
        except Exception as e:
            logger.error(f'Plotly histogram failed: {str(e)}')
            return -1

        # Generate year ticks for x-axis
        unique_years = sorted(df_exploded['year'].dropna().unique())
        # Set x-axis ticks to show only years
        year_positions = []
        year_labels = []
        for year in unique_years:
            # Find the first year-month for each year
            first_ym = video_counts[video_counts['year'] == year]['year_month'].min()
            if first_ym:
                year_positions.append(first_ym)
                year_labels.append(str(int(year)))

        # Update layout
        fig.update_layout(
            template='plotly',
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            font=dict(family=font_family or 'Arial', size=font_size or 12),
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
            bargap=0.1,
            xaxis=dict(
                tickmode='array',
                tickvals=year_positions,
                ticktext=year_labels,
                tickangle=45,
                title=dict(font=dict(size=(font_size or 12) + 10)),
                tickfont=dict(size=(font_size or 12) + 4)
                ),
            yaxis=dict(
                title=dict(font=dict(size=(font_size or 12) + 10)),
                tickfont=dict(size=(font_size or 12) + 4)
                )
            )

        # Save or show
        if save_file:
            if not file_name:
                file_name = 'hist_months'
            try:
                io_class.save_plotly_figure(fig, file_name)
            except Exception as e:
                logger.error(f'Failed to save plot: {str(e)}')
        else:
            fig.show()

        return fig

    def violin_plot(self, data_index, name, min_threshold, max_threshold, df_mapping, color=None,
                    pretty_text=False, xaxis_title=None, yaxis_title=None,
                    file_name=None, save_file=False, save_final=False, fig_save_width=1320,
                    fig_save_height=680, font_family=None, font_size=None, vlines=None, xrange=None,
                    data_file=None):
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
            file_name (str, optional): Name for saving the figure file. Defaults to None.
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
        with open(data_file, 'rb') as file:  # type: ignore
            data_tuple = pickle.load(file)

        # Extract the nested dictionary at the given index
        nested_dict = data_tuple[data_index]

        values = {}

        # Loop through each (city, lat, ...) key and collect values grouped by country
        for city_lat_long_cond, inner_dict in nested_dict.items():
            city, lat, _, _ = city_lat_long_cond.split("_")
            # Get country from mapping DataFrame
            country = metadata_class.get_value(df_mapping, "city", city, "lat", float(lat), "country")
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

    def bar(self, df, y: list, y_legend=None, x=None, stacked=False, pretty_text=False, orientation='v',
            xaxis_title=None, yaxis_title=None, show_all_xticks=False, show_all_yticks=False, show_text_labels=False,
            font_family=None, font_size=None, name_file=None, save_file=False, save_final=False, fig_save_width=1320,
            fig_save_height=680):
        """
        Barplot for questionnaire data. Passing a list with one variable will output a simple barplot; passing a list
        of variables will output a grouped barplot.

        Args:
            df (dataframe): dataframe with stimuli data.
            y (list): column names of dataframe to plot.
            y_legend (list, optional): names for variables to be shown in the legend.
            x (list): values in index of dataframe to plot for. If no value is given, the index of df is used.
            stacked (bool, optional): show as stacked chart.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            orientation (str, optional): orientation of bars. v=vertical, h=horizontal.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            show_all_xticks (bool, optional): show all ticks on x axis.
            show_all_yticks (bool, optional): show all ticks on y axis.
            show_text_labels (bool, optional): output automatically positioned text labels.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        logger.info('Creating bar chart for x={} and y={}.', x, y)
        # prettify text
        if pretty_text:
            for variable in y:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()
        # use index of df if no is given
        if x is None or (isinstance(x, (list, pd.Series, np.ndarray)) and len(x) == 0):
            x = df.index
        # create figure
        fig = go.Figure()

        # go over variables to plot
        for variable in range(len(y)):
            # showing text labels
            if show_text_labels:
                text = df[y[variable]]
            else:
                text = None

            # custom labels for legend
            if y_legend:
                name = y_legend[variable]
            else:
                name = y[variable]

            # plot variable
            fig.add_trace(go.Bar(x=x,
                                 y=df[y[variable]],
                                 name=name,
                                 orientation=orientation,
                                 text=text,
                                 textposition='auto'))
        # add tabs if multiple variables are plotted
        if len(y) > 1:
            fig.update_layout(barmode='group')
            buttons = list([dict(label='All',
                                 method='update',
                                 args=[{'visible': [True] * df[y].shape[0]},
                                       {'title': 'All', 'showlegend': True}])])
            # counter for traversing through stimuli
            counter_rows = 0
            for variable in y:
                visibility = [[counter_rows == j] for j in range(len(y))]
                visibility = [item for sublist in visibility for item in sublist]  # type: ignore
                button = dict(label=variable,
                              method='update',
                              args=[{'visible': visibility},
                                    {'title': variable}])
                buttons.append(button)
                counter_rows = counter_rows + 1
            updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
            fig['layout']['updatemenus'] = updatemenus  # pyright: ignore[reportIndexIssue]
            fig['layout']['title'] = 'All'  # pyright: ignore[reportIndexIssue]

        # update layout
        fig.update_layout(template=common.get_configs('plotly_template'), xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)

        # format text labels
        if show_text_labels:
            fig.update_traces(texttemplate='%{text:.2f}')

        # show all ticks on x axis
        if show_all_xticks:
            fig.update_layout(xaxis=dict(dtick=1))

        # show all ticks on x axis
        if show_all_yticks:
            fig.update_layout(yaxis=dict(dtick=1))

        # stacked bar chart
        if stacked:
            fig.update_layout(barmode='stack')

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

        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'bar_' + '-'.join(str(val) for val in y) + '_' + '-'.join(str(val) for val in x)

            io_class.save_plotly_figure(fig,
                                        name_file,
                                        width=fig_save_width,
                                        height=fig_save_height,
                                        save_final=save_final)  # also save as "final" figure

        # open it in localhost instead
        else:
            fig.show()
