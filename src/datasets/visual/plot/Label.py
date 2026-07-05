from datasets.visual.plot.LabelFit import LabelFit
from api.utils_future import ColorUtils, timer


class Label:
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")

    @staticmethod
    def _measure_text_size(ax, fig, text, fontsize):
        temp_text = ax.text(0, 0, text, fontsize=fontsize)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = temp_text.get_window_extent(renderer=renderer)
        temp_text.remove()
        return bbox.width, bbox.height, renderer

    @staticmethod
    def _rect_to_pixel_dims(ax, renderer, rect_w, rect_h):
        axes_bbox = ax.get_window_extent(renderer=renderer)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        rect_width_px = axes_bbox.width * rect_w / (xlim[1] - xlim[0])
        rect_height_px = axes_bbox.height * rect_h / (ylim[1] - ylim[0])
        return rect_width_px, rect_height_px

    @staticmethod
    @timer
    def _fit_fontsize(text, rect_w, rect_h, ax, fig):
        provisional_fontsize = 12
        text_w_px, text_h_px, renderer = Label._measure_text_size(
            ax, fig, text, provisional_fontsize
        )
        rect_w_px, rect_h_px = Label._rect_to_pixel_dims(
            ax, renderer, rect_w, rect_h
        )
        width_scale = rect_w_px / text_w_px if text_w_px > 0 else 1.0
        height_scale = rect_h_px / text_h_px if text_h_px > 0 else 1.0
        scale = min(width_scale, height_scale) * 0.95
        return max(1, provisional_fontsize * scale) * 0.7

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
