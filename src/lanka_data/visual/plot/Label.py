from lanka_data.visual.plot.LabelFit import LabelFit
from utils_future import ColorUtils, timer


class Label:
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")

    @staticmethod
    def _fit_fontsize(text, rect_w, rect_h, ax, fig):
        """
        Calculate font size to fit text within a rectangle.

        Args:
            text: String to fit
            rect_w: Rectangle width (in data coordinates)
            rect_h: Rectangle height (in data coordinates)
            ax: matplotlib axes object
            fig: matplotlib figure object

        Returns:
            Font size in points
        """
        # Start with provisional font size
        provisional_fontsize = 12

        # Create temporary text object
        temp_text = ax.text(0, 0, text, fontsize=provisional_fontsize)

        # Render to get accurate bounding box measurements
        fig.canvas.draw()

        # Get bounding box in display (pixel) coordinates
        renderer = fig.canvas.get_renderer()
        bbox = temp_text.get_window_extent(renderer=renderer)
        text_width_px = bbox.width
        text_height_px = bbox.height

        # Get axes bounding box in display coordinates
        axes_bbox = ax.get_window_extent(renderer=renderer)
        axes_width_px = axes_bbox.width
        axes_height_px = axes_bbox.height

        # Get data coordinate limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Convert rectangle dimensions from data coordinates to pixels
        rect_width_px = axes_width_px * rect_w / (xlim[1] - xlim[0])
        rect_height_px = axes_height_px * rect_h / (ylim[1] - ylim[0])

        # Calculate scale factors needed to fit text in rectangle
        width_scale = (
            rect_width_px / text_width_px if text_width_px > 0 else 1.0
        )
        height_scale = (
            rect_height_px / text_height_px if text_height_px > 0 else 1.0
        )

        # Use the limiting scale factor with safety margin
        scale = min(width_scale, height_scale) * 0.95

        # Calculate final font size
        final_fontsize = max(1, provisional_fontsize * scale)

        # Clean up temporary text object
        temp_text.remove()

        return final_fontsize * 0.7

    @classmethod
    @timer
    def draw(cls, gdf_region, ax):
        fig = ax.get_figure()
        for _, row in gdf_region.iterrows():
            cx, cy, rect_w, rect_h, angle_deg = LabelFit.best_label_fit(
                row.geometry
            )
            bg_color = row.get("color") or "white"
            text_color = "#666" if cls.IS_LIGHT_COLOR(bg_color) else "#aaa"
            label = (
                f'{row.get("region_name")}'
                if row.get("region_name")
                else str(row.get("region_id"))
            )

            if rect_w >= rect_h:
                text_w, text_h = rect_w, rect_h
                text_angle = angle_deg
            else:
                text_w, text_h = rect_h, rect_w
                text_angle = angle_deg + 90.0

            while text_angle > 90.0:
                text_angle -= 180.0

            fontsize = cls._fit_fontsize(label, text_w, text_h, ax, fig)
            ax.annotate(
                label,
                xy=(cx, cy),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
                rotation=text_angle,
            )
