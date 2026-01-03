import warnings
import common
import numpy as np
import plotly.express as px
from scipy.spatial import KDTree
from custom_logger import CustomLogger
from utils.plotting.layout import Layout
from utils.plotting.io import IO

layout_class = Layout()
plots_io_class = IO()
logger = CustomLogger(__name__)  # use custom logger


class Bivariate:
    def __init__(self) -> None:
        pass

    def scatter(self, df, x, y, extension=None, color=None, symbol=None, size=None, text=None, trendline=None,
                hover_data=None, marker_size=None, pretty_text=False, marginal_x='violin', marginal_y='violin',
                xaxis_title=None, yaxis_title=None, xaxis_range=None, yaxis_range=None, file_name=None,
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
            file_name (str, optional): name of file to save.
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
                    logger.warning("Column to be used as text not found, skipping display_label logic.")

            # scatter plot with histograms
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                fig = px.scatter(df,
                                 x=x,
                                 y=y,
                                 render_mode="svg",
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
                fig.update_traces(textposition=layout_class.improve_text_position(df[x]))

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
                if not file_name:
                    if extension is not None:
                        file_name = 'scatter_' + x + '-' + y + '-' + extension
                    else:
                        file_name = 'scatter_' + x + '-' + y

                # Final adjustments and display
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                plots_io_class.save_plotly_figure(fig, file_name, save_final=True)
            # open it in localhost instead
            else:
                fig.show()
