class Layout:
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
