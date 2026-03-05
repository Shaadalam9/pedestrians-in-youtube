import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import common
from utils.plotting.io import IO

io_class = IO()


class Maps:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _default_city_image_items() -> list[dict]:
        """Default overlay pack used for screenshot tiles and arrows.

        Notes
        -----
        * x/y/x_label/y_label are in paper coordinates (0..1).
        * approx_lon/approx_lat are used as a line anchor on the map.
        """

        return [
            {
                "city": "Tokyo",
                "country": "Japan",
                "file": "tokyo.png",
                "x": 0.933,
                "y": 0.58,
                "approx_lon": 165.2,
                "approx_lat": 7.2,
                "label": "Tokyo, Japan",
                "x_label": 0.982,
                "y_label": 0.641,
                "video": "oDejyTLYUTE",
                "x_video": 0.9305,
                "y_video": 0.521,
            },
            {
                "city": "Nairobi",
                "country": "Kenya",
                "file": "nairobi.png",
                "x": 0.72,
                "y": 0.38,
                "approx_lon": 70.2,
                "approx_lat": -20.0,
                "label": "Nairobi, Kenya",
                "x_label": 0.7695,
                "y_label": 0.442,
                "video": "VNLqnwoJqmM",
                "x_video": 0.72129,
                "y_video": 0.311,
            },
            {
                "city": "Los Angeles",
                "country": "United States",
                "file": "los_angeles.png",
                "x": 0.12,
                "y": 0.5,
                "approx_lon": -121.7,
                "approx_lat": 0.0,
                "label": "Los Angeles, CA, USA",
                "x_label": 0.0705,
                "y_label": 0.562,
                "video": "4uhMg5na888",
                "x_video": 0.126,
                "y_video": 0.44,
            },
            {
                "city": "Paris",
                "country": "France",
                "file": "paris.png",
                "x": 0.3915,
                "y": 0.68,
                "approx_lon": -30.6,
                "approx_lat": 30.4,
                "label": "Paris, France",
                "x_label": 0.366,
                "y_label": 0.752,
                "video": "ZTmjk8mSCq8",
                "x_video": 0.4171,
                "y_video": 0.62,
            },
            {
                "city": "Rio de Janeiro",
                "country": "Brazil",
                "file": "rio_de_janeiro.png",
                "x": 0.47,
                "y": 0.2,
                "approx_lon": -1.8,
                "approx_lat": -60.2,
                "label": "Rio de Janeiro, Brazil",
                "x_label": 0.4815,
                "y_label": 0.25,
                "video": "q83bl_GcsCo",
                "x_video": 0.441,
                "y_video": 0.131,
            },
            {
                "city": "Melbourne",
                "country": "Australia",
                "file": "melbourne.png",
                "x": 0.74,
                "y": 0.22,
                "approx_lon": 90.0,
                "approx_lat": -52.0,
                "label": "Melbourne, Australia",
                "x_label": 0.7893,
                "y_label": 0.27,
                "video": "gQ-9mmnfJjE",
                "x_video": 0.733,
                "y_video": 0.151,
            },
            # YOLO example tile (no arrow)
            {
                "file": "new_york_yolo.png",
                "x": 0.2,
                "y": 0.25,
                "sizex": 0.2,
                "sizey": 0.2,
                "label": "Example of YOLO output (New York, NY, USA)",
                "x_label": 0.101,
                "y_label": 0.361,
                "video": "Wyg213IZDI",
                "x_video": 0.258,
                "y_video": 0.131,
            },
        ]

    @staticmethod
    def _safe_open_image(path: str):
        """Open an image for Plotly embedding, return None if missing."""
        try:
            return Image.open(path)
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def mapbox_map(self, df, density_col=None, density_radius=30, hover_data=None, hover_name=None,
                   marker_size=5, file_name="mapbox_map", save_final=True):
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
            hover_name (list, optional): title on top of hover popup.
            marker_size (int, optional): size of markers.
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
                hover_name=hover_name,
                color=df["continent"],
                zoom=1.3  # pyright: ignore[reportArgumentType]
            )

            # Apply marker size
            fig.update_traces(marker=dict(size=marker_size))

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
                hover_data=hover_data,
                hover_name=hover_name
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
        io_class.save_plotly_figure(fig, file_name, save_final=save_final)

    def mapbox_map_footage(self, df: pd.DataFrame, *,
                           footage_col: str = "total_time",
                           hover_data=None,
                           hover_name: str | None = None,
                           marker_size: int = 3,
                           color_scale: str | list = "Viridis_r",
                           show_colorbar: bool = True,
                           colorbar_title: str | None = None,
                           log_colour: bool = True,
                           cap_quantile: float = 0.99,
                           range_q_low: float = 0.02,
                           range_q_high: float = 0.98,
                           show_images: bool = False,
                           image_items: list[dict] | None = None,
                           screenshots_dir: str | None = None,
                           file_name: str = "mapbox_map_footage",
                           save_final: bool = True):
        """Scatter map of cities coloured by footage amount using a continuous hue based scale.

        This colours markers by footage using a rainbow style colourscale (hue changes rather than lightness),
        so values remain visible without looking like a light to dark gradient.

        Requirements
        ------------
        df must contain at least: lat, lon and `footage_col`.
        """

        if df is None or len(df) == 0:
            return

        if screenshots_dir is None:
            screenshots_dir = os.path.join(common.root_dir, "screenshots")

        df_plot = df.copy()
        if footage_col not in df_plot.columns:
            raise KeyError(f"Missing column: {footage_col}")

        # Basic cleaning
        df_plot = df_plot.dropna(subset=["lat", "lon"]).copy()

        # Convert seconds to hours (if already hours, pass an hours column as footage_col)
        df_plot["footage_hours"] = pd.to_numeric(df_plot[footage_col], errors="coerce") / 3600.0
        df_plot["footage_hours"] = df_plot["footage_hours"].fillna(0.0)

        # Colour coding: clip outliers then apply log1p to spread smaller values
        x = df_plot["footage_hours"].clip(lower=0.0)

        # Cap extreme values so a few very large cities do not flatten the colour range
        cap = x.quantile(cap_quantile) if len(x) else 0.0
        if pd.isna(cap) or cap <= 0:
            cap = float(x.max()) if len(x) else 0.0
        x_capped = x.clip(upper=cap)

        # log1p keeps zero well defined
        df_plot["footage_log_capped"] = np.log1p(x_capped)

        colour_col = "footage_log_capped" if log_colour else "footage_hours"

        # Robust colour range for better contrast
        if log_colour:
            q = df_plot[colour_col].quantile([range_q_low, range_q_high])
            cmin = float(q.iloc[0]) if len(q) else float(df_plot[colour_col].min())
            cmax = float(q.iloc[1]) if len(q) else float(df_plot[colour_col].max())
            if (not np.isfinite(cmin)) or (not np.isfinite(cmax)) or cmin == cmax:
                cmin = float(df_plot[colour_col].min())
                cmax = float(df_plot[colour_col].max())
        else:
            cmin = float(df_plot[colour_col].min())
            cmax = float(df_plot[colour_col].max())

        if cmin == cmax:
            cmax = cmin + 1e-9

        # Ensure footage appears on hover
        hover_data_local = hover_data
        if hover_data_local is None:
            hover_data_local = ["footage_hours"]
        elif isinstance(hover_data_local, list):
            if "footage_hours" not in hover_data_local:
                hover_data_local = list(hover_data_local) + ["footage_hours"]
        elif isinstance(hover_data_local, dict):
            hover_data_local = dict(hover_data_local)
            hover_data_local.setdefault("footage_hours", True)

        if log_colour:
            if isinstance(hover_data_local, list):
                if "footage_log_capped" not in hover_data_local:
                    hover_data_local = list(hover_data_local) + ["footage_log_capped"]
            elif isinstance(hover_data_local, dict):
                hover_data_local.setdefault("footage_log_capped", True)

        if colorbar_title is None:
            colorbar_title = "log1p(capped footage hours)" if log_colour else "Footage (hours)"

        fig = px.scatter_map(
            df_plot,
            lat="lat",
            lon="lon",
            hover_data=hover_data_local,
            hover_name=hover_name,
            color=colour_col,
            color_continuous_scale=color_scale,
            range_color=(cmin, cmax),
            zoom=1.3,  # pyright: ignore[reportArgumentType]
            map_style="carto-positron",
        )

        # Main points
        fig.update_traces(marker=dict(size=marker_size, opacity=0.95), selector=dict(type="scattermap"))

        # Black halo layer underneath, to keep points readable on any background
        halo_size = max(marker_size + 2, 2)
        outline_trace = go.Scattermap(
            lat=df_plot["lat"],
            lon=df_plot["lon"],
            mode="markers",
            marker=dict(size=halo_size, color="black", opacity=0.75),
            hoverinfo="skip",
            showlegend=False,
            subplot="map",
            name="_outline",
        )
        fig.add_trace(outline_trace)

        # Reorder traces so halo is drawn first (underneath). Plotly only allows permutations.
        data_list = list(fig.data)
        # last trace is the halo we just added
        data_list = [data_list[-1]] + data_list[:-1]
        fig.data = tuple(data_list)

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(
                family=common.get_configs("font_family"),
                size=common.get_configs("font_size"),
            ),
            showlegend=False,
        )

        fig.update_coloraxes(
            showscale=show_colorbar,
            colorbar=dict(
                title=colorbar_title,
                x=0.5,
                xanchor="center",
                y=0.02 if not show_images else 0.01,
                yanchor="bottom",
                orientation="h",
                len=0.55,
                thickness=10,
                bgcolor="rgba(255,255,255,0.85)",
            ) if show_colorbar else {}
        )

        # ------------------------------
        # Optional screenshot overlays
        # ------------------------------
        if show_images:
            if image_items is None:
                image_items = self._default_city_image_items()

            # Images and labels (paper coordinates)
            for item in image_items:
                file_name_img = item.get("file")
                if file_name_img:
                    img_path = os.path.join(screenshots_dir, file_name_img)
                    img = self._safe_open_image(img_path)
                    if img is not None:
                        fig.add_layout_image(
                            dict(
                                source=img,
                                xref="paper",
                                yref="paper",
                                x=item.get("x", 0.5),
                                y=item.get("y", 0.5),
                                sizex=item.get("sizex", 0.1),
                                sizey=item.get("sizey", 0.1),
                                xanchor=item.get("xanchor", "center"),
                                yanchor=item.get("yanchor", "middle"),
                                layer=item.get("layer", "above"),
                            )
                        )

                label = item.get("label")
                if label:
                    fig.add_annotation(
                        text=label,
                        x=item.get("x_label", item.get("x", 0.5)),
                        y=item.get("y_label", item.get("y", 0.5) + 0.1),
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black",
                        borderwidth=1,
                    )

                video = item.get("video")
                if video:
                    fig.add_annotation(
                        text=video,
                        x=item.get("x_video", item.get("x", 0.5)),
                        y=item.get("y_video", item.get("y", 0.5) - 0.1),
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        align="center",
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black",
                        borderwidth=1,
                    )

            # Arrows from tile anchor to actual city coordinates
            if {"city", "country", "lat", "lon"}.issubset(df_plot.columns):
                for item in image_items:
                    if not ("city" in item and "country" in item):
                        continue
                    if not ("approx_lon" in item and "approx_lat" in item):
                        continue

                    city = str(item["city"]).strip().lower()
                    country = str(item["country"]).strip().lower()

                    rows = df_plot[
                        (df_plot["city"].astype(str).str.strip().str.lower() == city)
                        & (df_plot["country"].astype(str).str.strip().str.lower() == country)
                    ]
                    if rows.empty:
                        continue

                    lon_city = float(rows["lon"].iloc[0])
                    lat_city = float(rows["lat"].iloc[0])

                    fig.add_trace(
                        go.Scattermap(
                            lon=[float(item["approx_lon"]), lon_city],
                            lat=[float(item["approx_lat"]), lat_city],
                            mode="lines",
                            line=dict(width=2, color="black"),
                            hoverinfo="skip",
                            showlegend=False,
                            subplot="map",
                        )
                    )

        io_class.save_plotly_figure(fig, file_name, save_final=save_final)

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
        io_class.save_plotly_figure(fig, "world_map", save_final=True)

    def map_world(self, df, *, color, title=None, projection="natural earth", hover_name="country", hover_data=None,
                  show_colorbar=False, colorbar_title=None, colorbar_kwargs=None, color_scale="YlOrRd",
                  show_cities=False, df_cities=None, city_marker_size=3, show_images=False, image_items=None,
                  denmark_greenland=False, save_file=False, save_final=False, file_name="map",
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
            file_name (str): Base filename without extension.
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
        print(type(df_cities))
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
            io_class.save_plotly_figure(fig, file_name, save_final=save_final)
        else:
            fig.show()

        return fig
