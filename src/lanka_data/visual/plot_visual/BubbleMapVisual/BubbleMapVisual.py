from lanka_data.visual.plot.color_spec.ColorSpecFactory import ColorSpecFactory
from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot.map.BubbleData.BubbleData import BubbleData
from lanka_data.visual.plot.map.GeoData.GeoData import GeoData
from lanka_data.visual.plot.map.RegionPopulationFilter import \
    RegionPopulationFilter
from lanka_data.visual.plot_visual.BubbleMapVisual.BubbleMapDrawMixin import \
    BubbleMapDrawMixin
from lanka_data.visual.plot_visual.BubbleMapVisual.BubbleMapLabelMixin import \
    BubbleMapLabelMixin
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual
from utils_future import timer


class BubbleMapVisual(PlotVisual, BubbleMapDrawMixin, BubbleMapLabelMixin):
    @classmethod
    def get_description(cls):
        return (
            "Renders data as a map with bubble markers sized by values and "
            "colored by categories"
        )

    @staticmethod
    def _region_to_name(data_list):
        return {
            d["region_id"]: d.get("region_name") or str(d["region_id"])
            for d in data_list
        }

    @timer
    def draw(self, dataset, fig):
        region_color_map, value_to_color, value_to_region = (
            ColorSpecFactory.get_color_spec(dataset, self.how_cmd).unpack()
        )
        data_list_all = dataset.get_data_table()
        gdf_region = GeoData.get_geopandas_dataframe(
            data_list_all, False
        ).copy()
        data_list = RegionPopulationFilter.filter(data_list_all)
        layout = BubbleData.get_bubble_layout(data_list)
        region_to_name = self._region_to_name(data_list)

        gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])

        self._draw_background(ax, gdf_region)
        self._draw_bubbles(ax, layout, region_color_map)
        self._draw_labels(
            ax,
            layout,
            region_to_name,
            region_color_map,
            len(region_to_name),
        )
        Legend.draw(
            value_to_color, legend_ax, value_to_region=value_to_region
        )
        ax.set_axis_off()
