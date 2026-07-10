from lanka_data.visual.plot.color_spec.ColorSpecFactory import ColorSpecFactory
from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot.map.RegionPopulationFilter import \
    RegionPopulationFilter
from lanka_data.visual.plot.map.SquareData.SquareData import SquareData
from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual
from lanka_data.visual.plot_visual.SquareMapVisual.SquareMapBoundaryMixin \
    import SquareMapBoundaryMixin
from lanka_data.visual.plot_visual.SquareMapVisual.SquareMapDrawMixin import \
    SquareMapDrawMixin
from lanka_data.visual.plot_visual.SquareMapVisual.SquareMapLabelMixin import \
    SquareMapLabelMixin
from utils_future import timer


class SquareMapVisual(
    PlotVisual,
    SquareMapDrawMixin,
    SquareMapLabelMixin,
    SquareMapBoundaryMixin,
):
    @staticmethod
    def _region_to_name(data_list):
        return {
            d["region_id"]: d.get("region_name") or str(d["region_id"])
            for d in data_list
        }

    @staticmethod
    def _scale_text(value_min, value_max):
        Q = 100
        a = round(value_min / Q) * Q
        b = round(value_max / Q) * Q
        if a == b:
            return f"Square = {a:,} people"

        mid = round((a + b) / 2 / Q) * Q
        span = round((b - a) / 2 / Q) * Q
        return f"Square = {mid:,} ± {span:,} people"

    @classmethod
    def _draw_scale(cls, ax, layout):
        value_min = layout.get("value_per_square_min")
        value_max = layout.get("value_per_square_max")
        if value_min is None or value_max is None:
            return
        fig = ax.get_figure()
        fig.text(
            0.5,
            0.085,
            cls._scale_text(value_min, value_max),
            fontsize=Style.FONT_SIZE_METADATA,
            ha="center",
            va="bottom",
            color=Style.COLOR_METADATA,
        )

    @timer
    def draw(self, dataset, fig):
        region_color_map, value_to_color, value_to_region = (
            ColorSpecFactory.get_color_spec(dataset, self.how_cmd).unpack()
        )
        data_list = RegionPopulationFilter.filter(dataset.get_data_table())
        layout = SquareData.get_square_layout(data_list)
        region_to_name = self._region_to_name(data_list)

        gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])

        self._draw_squares(ax, layout, region_color_map)
        self._draw_boundaries(ax, layout)
        self._draw_labels(
            ax,
            layout,
            region_to_name,
            region_color_map,
            len(region_to_name),
        )
        self._draw_scale(ax, layout)
        Legend.draw(
            value_to_color, legend_ax, value_to_region=value_to_region
        )
        ax.set_axis_off()
