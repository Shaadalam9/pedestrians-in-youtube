import os
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import common
from utils.plotting.io import IO

io_class = IO()


class Maps:
    def __init__(self) -> None:
        pass

    def mapbox_map(self, df, density_col=None, density_radius=30, hover_data=None, hover_name=None,
                   file_name="mapbox_map", save_final=True):
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
