from lanka_data.visual.plot.Label import Label
from lanka_data.visual.plot.map.GeoData import GeoData
from lanka_data.visual.plot_visual.BivariateMapVisual.BivariateData import \
    BivariateData
from lanka_data.visual.plot_visual.BivariateMapVisual.BivariatePalette import \
    BivariatePalette
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual


class BivariateMapBaseVisual(PlotVisual):
    N_BINS = 3
    EDGE_COLOR = "#fff"
    EDGE_WIDTH = 0.2

    @property
    def palette(self):
        return BivariatePalette(self.N_BINS)

    @property
    def categories(self):
        return self.command.how.categories

    @property
    def measure_labels(self):
        whats = self.command.what.whats
        first = whats[0]
        last = whats[-1] if len(whats) > 1 else whats[0]
        categories = self.categories
        if categories:
            first_category, last_category = BivariateData._pair(categories)
            if first_category:
                first = f"{first_category} ({first})"
            if last_category:
                last = f"{last_category} ({last})"
        return first, last

    def _classified_points(self, dataset):
        points = BivariateData.points(
            dataset.get_data_table(), self.categories
        )
        return BivariateData.classify(points, self.N_BINS)

    def _region_color_map(self, points):
        palette = self.palette
        return {
            point["region_id"]: palette.color(point["x_bin"], point["y_bin"])
            for point in points
        }

    def _get_gdf_region(self, dataset, region_color_map):
        gdf_region = GeoData.get_geopandas_dataframe(
            dataset.get_data_table(), False
        ).copy()
        gdf_region["color"] = (
            gdf_region["region_id"]
            .map(region_color_map)
            .fillna(BivariatePalette.NEUTRAL)
        )
        return gdf_region

    def _draw_map(self, ax, gdf_region):
        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor=self.EDGE_COLOR,
            linewidth=self.EDGE_WIDTH,
        )
        Label.draw(gdf_region, ax, len(gdf_region))
        ax.set_axis_off()
