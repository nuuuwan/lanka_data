from lanka_data.visual.plot.color_spec import ColorSpecFactory
from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot.map.HexData import HexData
from lanka_data.visual.plot_visual.HexMapVisual.HexMapBoundaryMixin import (
    HexMapBoundaryMixin,
)
from lanka_data.visual.plot_visual.HexMapVisual.HexMapDrawMixin import (
    HexMapDrawMixin,
)
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual
from utils_future import timer


class HexMapVisual(PlotVisual, HexMapDrawMixin, HexMapBoundaryMixin):
    MAX_REGIONS_TO_LABEL = 30

    @staticmethod
    def _region_to_name(data_list):
        return {
            d["region_id"]: d.get("region_name") or str(d["region_id"])
            for d in data_list
        }

    @staticmethod
    def _draw_scale(ax, layout):
        value_per_hex = layout.get("value_per_hex")
        if not value_per_hex:
            return
        ax.set_title(
            f"Each hexagon represents ~{round(value_per_hex):,} people",
            fontsize=9,
        )

    @timer
    def draw(self, dataset, fig):
        region_color_map, value_to_color = ColorSpecFactory.get_color_spec(
            dataset, self.how_cmd
        ).unpack()
        data_list = dataset.get_data_table()
        layout = HexData.get_hex_layout(data_list)
        region_to_name = self._region_to_name(data_list)

        gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])

        self._draw_hexes(ax, layout, region_color_map)
        self._draw_boundaries(ax, layout)
        if len(region_to_name) <= self.MAX_REGIONS_TO_LABEL:
            self._draw_labels(ax, layout, region_to_name, region_color_map)
        self._draw_scale(ax, layout)
        Legend.draw(value_to_color, legend_ax)
        ax.set_axis_off()
