from lanka_data.visual.plot.color_spec import ColorSpecFactory
from lanka_data.visual.plot.Label import Label
from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot.map.GeoData import GeoData
from lanka_data.visual.plot.map.GeoData.DistrictBackgroundGeoData import (
    DistrictBackgroundGeoData,
)
from lanka_data.visual.plot.map.RegionPopulationFilter import (
    RegionPopulationFilter,
)
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual
from utils_future import timer


class MapVisual(PlotVisual):
    DEFAULT_EDGE_COLOR = "#fff"
    DEFAULT_EDGE_WIDTH = 0.2
    LINE_WIDTH = 0.8
    LINE_GEOM_TYPES = ["LineString", "MultiLineString"]
    BACKGROUND_COLOR = "#eee"
    BACKGROUND_EDGE_COLOR = "#ddd"
    BACKGROUND_EDGE_WIDTH = 0.4

    def _get_gdf_region(self, dataset, region_color_map):
        data_list = dataset.get_data_table()
        if not data_list:
            return None
        is_cartogram = "Cartogram" in self.how_cmd
        if is_cartogram:
            data_list = RegionPopulationFilter.filter(data_list)
        gdf_region = GeoData.get_geopandas_dataframe(
            data_list, is_cartogram
        ).copy()
        gdf_region["color"] = gdf_region["region_id"].map(region_color_map)
        return gdf_region

    def _is_line_region(self, gdf_region):
        return gdf_region.geom_type.isin(self.LINE_GEOM_TYPES).any()

    def _draw_district_background(self, ax):
        gdf_districts = DistrictBackgroundGeoData.get()
        if gdf_districts is None or gdf_districts.empty:
            return
        gdf_districts.plot(
            ax=ax,
            color=self.BACKGROUND_COLOR,
            edgecolor=self.BACKGROUND_EDGE_COLOR,
            linewidth=self.BACKGROUND_EDGE_WIDTH,
        )

    def _draw_region(self, gdf_region, ax):
        if self._is_line_region(gdf_region):
            self._draw_district_background(ax)
            gdf_region.plot(
                ax=ax,
                color=gdf_region["color"],
                linewidth=self.LINE_WIDTH,
            )
            return
        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor=self.DEFAULT_EDGE_COLOR,
            linewidth=self.DEFAULT_EDGE_WIDTH,
        )
        Label.draw(gdf_region, ax, len(gdf_region))

    @timer
    def draw(self, dataset, fig):
        region_color_map, value_to_color, value_to_region = (
            ColorSpecFactory.get_color_spec(dataset, self.how_cmd).unpack()
        )

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

        if gdf_region is not None:
            self._draw_region(gdf_region, ax)
        Legend.draw(
            value_to_color, legend_ax, value_to_region=value_to_region
        )
        ax.set_axis_off()
