from lanka_data.api.how.map.color_spec import ColorSpecFactory
from lanka_data.api.how.map.GeoData import GeoData
from lanka_data.api.how.plot.Label import Label
from lanka_data.api.how.plot.Legend import Legend
from lanka_data.api.how.plot.Text import Text
from lanka_data.api.what.DiffWhat import DiffWhat
from utils_future import Parse


class MapSubFigure:
    DEFAULT_EDGE_COLOR = "#fff"
    DEFAULT_EDGE_WIDTH = 0.2
    MAX_REGIONS_TO_LABEL = 30

    def __init__(self, figure_label, command, is_cartogram, subfigure):
        self.figure_label = figure_label
        self.command = command
        self.is_cartogram = is_cartogram
        self.subfigure = subfigure

    def draw(self):
        params = getattr(self.command.get_how(), "params", None)
        if isinstance(self.command.get_what(), DiffWhat) and params not in (
            "Flips",
            "Segregation",
        ):
            from lanka_data.api.how.plot.ChartSubFigure import ChartSubFigure

            return ChartSubFigure(
                self.figure_label,
                self.command,
                self.is_cartogram,
                self.subfigure,
            ).draw()

        how = self.command.get_how()
        what = self.command.get_what()
        when = self.command.get_when()
        where = self.command.get_where()

        result_data = how.get_data(what, when, where)
        data_list = result_data["data_list"]
        gdf_region = GeoData.get_geopandas_dataframe(
            data_list,
            self.is_cartogram,
        ).copy()
        region_color_map, value_to_color = ColorSpecFactory.get_color_spec(
            self.command
        ).unpack()
        gdf_region["color"] = gdf_region["region_id"].map(region_color_map)

        gs = self.subfigure.add_gridspec(
            1,
            2,
            width_ratios=[5, 1],
            wspace=0.05,
        )
        ax = self.subfigure.add_subplot(gs[0])
        legend_ax = self.subfigure.add_subplot(gs[1])

        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor=self.DEFAULT_EDGE_COLOR,
            linewidth=self.DEFAULT_EDGE_WIDTH,
        )
        skip_colors = {
            color
            for value, color in (value_to_color or {}).items()
            if value.startswith("(No Data)")
            or value.startswith("(No Flip)")
            or (
                Parse.float(value) is not None
                and abs(Parse.float(value)) < 0.001
            )
        }

        labeled_gdf = gdf_region[~gdf_region["color"].isin(skip_colors)]
        if len(labeled_gdf) < self.MAX_REGIONS_TO_LABEL:
            Label.draw(labeled_gdf, ax)
        Legend.draw(value_to_color, legend_ax)

        ax.set_axis_off()
        Text.plot(
            self.subfigure,
            (0.5, 0.9),
            self.figure_label,
            fontsize=16,
            color="#000",
        )
        return result_data
