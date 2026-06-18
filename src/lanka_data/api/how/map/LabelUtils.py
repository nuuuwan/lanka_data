from lanka_data.api.how.map.LabelFitUtils import LabelFitUtils
from utils_future import ColorUtils


class LabelUtils:
    @staticmethod
    def _fit_fontsize(text, rect_w, rect_h, ax, fig):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        frac_w = rect_w / (x_max - x_min)
        frac_h = rect_h / (y_max - y_min)
        root_fig = getattr(fig, "figure", fig)
        avail_w_pts = frac_w * root_fig.get_figwidth() * 72
        avail_h_pts = frac_h * root_fig.get_figheight() * 72
        n_chars = max(len(text), 1)
        size_from_w = avail_w_pts / (n_chars * 0.6)
        size_from_h = avail_h_pts / 1.2
        return min(size_from_w, size_from_h, 18) * 0.3

    @staticmethod
    def draw_labels(gdf_region, ax):
        fig = ax.get_figure()
        for _, row in gdf_region.iterrows():
            cx, cy, rect_w, rect_h, angle_deg = LabelFitUtils._best_label_fit(
                row.geometry
            )
            bg_color = row.get("color", "black")
            text_color = (
                "black" if ColorUtils._is_light_color(bg_color) else "white"
            )
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
            fontsize = LabelUtils._fit_fontsize(
                label, text_w, text_h, ax, fig
            )
            ax.annotate(
                label,
                xy=(cx, cy),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
                rotation=text_angle,
            )
