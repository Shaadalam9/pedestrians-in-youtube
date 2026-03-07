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
        * approx_lon/approx_lat place the screenshot tile on the map.
        * city/country are used to draw the connector line to the actual city.
        * label_dlon/label_dlat and video_dlon/video_dlat control text offsets
          from the tile centre.
        * file names are assumed to match the screenshots directory.
        """

        return [
            {
                "city": "Cape Town",
                "country": "South Africa",
                "file": "cape_town.jpg",
                "approx_lon": 30.0,
                "approx_lat": -48.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "Cape Town, South Africa",
                "label_dlon": -10.0,
                "label_dlat": 9.5,
                "label_font_size": 9,
                "video": "0xP7JgDiBb8",
                "video_dlon": 10.0,
                "video_dlat": -9.5,
                "video_font_size": 9,
            },
            {
                "city": "Seoul",
                "country": "South Korea",
                "file": "seoul.jpg",
                "approx_lon": 174.0,
                "approx_lat": 40.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "Seoul, South Korea",
                "label_dlon": -6.0,
                "label_dlat": 9.5,
                "label_font_size": 9,
                "video": "qOx5CwCrN9k",
                "video_dlon": 8.0,
                "video_dlat": -9.5,
                "video_font_size": 9,
            },
            {
                "city": "London",
                "country": "United Kingdom",
                "file": "london.jpg",
                "approx_lon": -32.0,
                "approx_lat": 50.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 6.0,
                "label": "London, UK",
                "label_dlon": -10.0,
                "label_dlat": 7.0,
                "label_font_size": 9,
                "video": "QI4_dGvZ5yE",
                "video_dlon": 10.0,
                "video_dlat": -7.0,
                "video_font_size": 9,
            },
            {
                "city": "New York",
                "country": "United States",
                "file": "new_york.jpg",
                "approx_lon": -42.0,
                "approx_lat": 28.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "New York, US",
                "label_dlon": -9.0,
                "label_dlat": 9.0,
                "label_font_size": 9,
                "video": "_Wyg213IZDI",
                "video_dlon": 9.0,
                "video_dlat": -9.0,
                "video_font_size": 9,
            },
            {
                "city": "Sydney",
                "country": "Australia",
                "file": "sydney.jpg",
                "approx_lon": 118.0,
                "approx_lat": -46.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "Sydney, Australia",
                "label_dlon": -7.5,
                "label_dlat": 9.5,
                "label_font_size": 9,
                "video": "wMu6Va5PhGY",
                "video_dlon": 9.0,
                "video_dlat": -9.5,
                "video_font_size": 9,
            },
            {
                "city": "Sao Paulo",
                "country": "Brazil",
                "file": "sao_paulo.jpg",
                "approx_lon": -20.0,
                "approx_lat": -45.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "Sao Paulo, Brazil",
                "label_dlon": -7.5,
                "label_dlat": 8.5,
                "label_font_size": 9,
                "video": "Ic2ERD7kt4o",
                "video_dlon": 10.0,
                "video_dlat": -8.5,
                "video_font_size": 9,
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

    @staticmethod
    def _rect_corners(center_lon: float,
                      center_lat: float,
                      half_width_deg: float,
                      half_height_deg: float):
        """Return geographic corners for a rectangular image layer."""
        return [
            [center_lon - half_width_deg, center_lat + half_height_deg],
            [center_lon + half_width_deg, center_lat + half_height_deg],
            [center_lon + half_width_deg, center_lat - half_height_deg],
            [center_lon - half_width_deg, center_lat - half_height_deg],
        ]

    @staticmethod
    def _tile_edge_point(
        center_lon: float,
        center_lat: float,
        target_lon: float,
        target_lat: float,
        half_width_deg: float,
        half_height_deg: float,
    ) -> tuple[float, float]:
        """Return the point where the line from the tile centre exits the tile rectangle."""
        dx = target_lon - center_lon
        dy = target_lat - center_lat

        if dx == 0 and dy == 0:
            return center_lon, center_lat

        tx = float("inf") if dx == 0 else half_width_deg / abs(dx)
        ty = float("inf") if dy == 0 else half_height_deg / abs(dy)
        t = min(tx, ty)

        edge_lon = center_lon + t * dx
        edge_lat = center_lat + t * dy
        return edge_lon, edge_lat

    @staticmethod
    def _point_geojson(lon: float, lat: float, text: str, property_name: str = "label") -> dict:
        """Create a tiny GeoJSON point feature collection for a map symbol layer."""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                    "properties": {
                        property_name: text,
                    },
                }
            ],
        }

    @staticmethod
    def _symbol_text_layer(*,
                           lon: float,
                           lat: float,
                           text: str,
                           font_size: int = 16,
                           text_color: str = "black",
                           property_name: str = "label") -> dict | None:
        """Return a crisp map-attached text layer using a GeoJSON symbol layer."""
        if not text:
            return None

        return dict(
            sourcetype="geojson",
            source=Maps._point_geojson(lon, lat, text, property_name=property_name),
            type="symbol",
            symbol=dict(
                text="{" + property_name + "}",
                placement="point",
                textfont=dict(
                    size=int(font_size),
                    color=text_color,
                ),
            ),
        )

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
        """Scatter map of cities coloured by footage amount using a continuous hue based scale."""

        if df is None or len(df) == 0:
            return

        if screenshots_dir is None:
            screenshots_dir = os.path.join(common.root_dir, "screenshots")

        df_plot = df.copy()
        if footage_col not in df_plot.columns:
            raise KeyError(f"Missing column: {footage_col}")

        df_plot = df_plot.dropna(subset=["lat", "lon"]).copy()

        df_plot["footage_hours"] = pd.to_numeric(df_plot[footage_col], errors="coerce") / 3600.0
        df_plot["footage_hours"] = df_plot["footage_hours"].fillna(0.0)

        x = df_plot["footage_hours"].clip(lower=0.0)

        cap = x.quantile(cap_quantile) if len(x) else 0.0
        if pd.isna(cap) or cap <= 0:
            cap = float(x.max()) if len(x) else 0.0
        x_capped = x.clip(upper=cap)

        df_plot["footage_log_capped"] = np.log1p(x_capped)

        colour_col = "footage_log_capped" if log_colour else "footage_hours"

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
            colorbar_title = "" if log_colour else "Footage (hours)"

        fig = px.scatter_map(
            df_plot,
            lat="lat",
            lon="lon",
            hover_data=hover_data_local,
            hover_name=hover_name,
            color=colour_col,
            color_continuous_scale=color_scale,
            range_color=(cmin, cmax),
            zoom=1.3,
            map_style="carto-positron",
        )

        fig.update_traces(marker=dict(size=marker_size, opacity=0.95), selector=dict(type="scattermap"))

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

        data_list = list(fig.data)
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

        if show_images:
            if image_items is None:
                image_items = self._default_city_image_items()

            def _item_anchor_lon(item: dict) -> float | None:
                value = item.get("img_lon", item.get("approx_lon"))
                return float(value) if value is not None else None

            def _item_anchor_lat(item: dict) -> float | None:
                value = item.get("img_lat", item.get("approx_lat"))
                return float(value) if value is not None else None

            def _item_image_corners(item: dict):
                lon0 = _item_anchor_lon(item)
                lat0 = _item_anchor_lat(item)
                if lon0 is None or lat0 is None:
                    return None

                half_width = float(item.get("img_half_width_deg", 10.0))
                half_height = float(item.get("img_half_height_deg", 5.0))

                return [
                    [lon0 - half_width, lat0 + half_height],
                    [lon0 + half_width, lat0 + half_height],
                    [lon0 + half_width, lat0 - half_height],
                    [lon0 - half_width, lat0 - half_height],
                ]

            map_layers = list(fig.layout.map.layers) if getattr(fig.layout.map, "layers", None) else []

            for item in image_items:
                file_name_img = item.get("file")
                corners = _item_image_corners(item)

                if file_name_img and corners is not None:
                    img_path = os.path.join(screenshots_dir, file_name_img)
                    img = self._safe_open_image(img_path)
                    if img is not None:
                        map_layers.append(
                            dict(
                                sourcetype="image",
                                source=img,
                                coordinates=corners,
                                opacity=float(item.get("opacity", 1.0)),
                                below=item.get("below", "traces"),
                            )
                        )
                    elif item.get("source") is not None:
                        map_layers.append(
                            dict(
                                sourcetype="image",
                                source=item["source"],
                                coordinates=corners,
                                opacity=float(item.get("opacity", 1.0)),
                                below=item.get("below", "traces"),
                            )
                        )

                label = item.get("label")
                anchor_lon = _item_anchor_lon(item)
                anchor_lat = _item_anchor_lat(item)

                if label and anchor_lon is not None and anchor_lat is not None:
                    fig.add_trace(
                        go.Scattermap(
                            lon=[anchor_lon + float(item.get("label_dlon", 0.0))],
                            lat=[anchor_lat + float(item.get("label_dlat", 6.0))],
                            mode="text",
                            text=[label],
                            textfont=dict(size=int(item.get("label_font_size", 12)), color="black"),
                            hoverinfo="skip",
                            showlegend=False,
                            subplot="map",
                        )
                    )

                video = item.get("video")
                if video and anchor_lon is not None and anchor_lat is not None:
                    fig.add_trace(
                        go.Scattermap(
                            lon=[anchor_lon + float(item.get("video_dlon", 0.0))],
                            lat=[anchor_lat + float(item.get("video_dlat", -6.0))],
                            mode="text",
                            text=[video],
                            textfont=dict(size=int(item.get("video_font_size", 10)), color="black"),
                            hoverinfo="skip",
                            showlegend=False,
                            subplot="map",
                        )
                    )

            if map_layers:
                fig.update_layout(map=dict(layers=map_layers))

            if {"city", "country", "lat", "lon"}.issubset(df_plot.columns):
                for item in image_items:
                    if not ("city" in item and "country" in item):
                        continue

                    anchor_lon = _item_anchor_lon(item)
                    anchor_lat = _item_anchor_lat(item)
                    if anchor_lon is None or anchor_lat is None:
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

                    half_w = float(item.get("img_half_width_deg", 10.0))
                    half_h = float(item.get("img_half_height_deg", 5.0))

                    edge_lon, edge_lat = self._tile_edge_point(
                        center_lon=anchor_lon,
                        center_lat=anchor_lat,
                        target_lon=lon_city,
                        target_lat=lat_city,
                        half_width_deg=half_w,
                        half_height_deg=half_h,
                    )

                    fig.add_trace(
                        go.Scattermap(
                            lon=[edge_lon, lon_city],
                            lat=[edge_lat, lat_city],
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
