from lanka_data.visual.plot.color_spec import ColorSpecFactory
from lanka_data.visual.plot.Label import Label
from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot.map.GeoData import GeoData
from lanka_data.visual.plot.Text import Text
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual
from utils_future import timer


class MapVisual(PlotVisual):
    DEFAULT_EDGE_COLOR = "#fff"
    DEFAULT_EDGE_WIDTH = 0.2
    MAX_REGIONS_TO_LABEL = 30

    def _get_gdf_region(self, dataset, region_color_map):
        data_list = dataset.get_data_table()
        gdf_region = GeoData.get_geopandas_dataframe(
            data_list, "Cartogram" in self.how_cmd
        ).copy()
        gdf_region["color"] = gdf_region["region_id"].map(region_color_map)
        return gdf_region

    @timer
    def draw(self, dataset, fig):
        region_color_map, value_to_color = ColorSpecFactory.get_color_spec(
            dataset, self.how_cmd
        ).unpack()

        gdf_region = self._get_gdf_region(
            dataset,
            region_color_map,
        )

        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[5, 1],
            wspace=0.05,
        )
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])

        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor=self.DEFAULT_EDGE_COLOR,
            linewidth=self.DEFAULT_EDGE_WIDTH,
        )
        if len(gdf_region) <= self.MAX_REGIONS_TO_LABEL:
            Label.draw(gdf_region, ax)
        Legend.draw(value_to_color, legend_ax)
        ax.set_axis_off()
        Text.plot(
            fig,
            (0.5, 0.9),
            dataset.get_year(),
            fontsize=16,
            color="#000",
        )
