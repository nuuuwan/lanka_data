from .BumpChartDataMixin import BumpChartDataMixin
from .BumpChartDrawMixin import BumpChartDrawMixin
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual


class BumpChartVisual(BumpChartDataMixin, BumpChartDrawMixin, PlotVisual):
    def draw(self, dataset, fig):
        subregions = self._build_subregions(dataset.get_data_table())
        when_cmd = getattr(self.command, "when_cmd", None)
        tokens = when_cmd.split("-") if when_cmd and "-" in when_cmd else []
        when_labels = tokens if len(tokens) == 2 else ["Start", "End"]
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        if not subregions:
            ax.set_axis_off()
            return
        if not self._has_diff_values(subregions):
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "BumpChart requires a change range, e.g. 2012-2024.",
                ha="center",
                va="center",
                fontsize=10,
                color="#444",
                transform=ax.transAxes,
            )
            return
        rank_map1, rank_map2 = self._get_rank_maps(subregions)
        region_ids = self._get_selected_region_ids(rank_map1, rank_map2)
        id_to_name = self._build_id_to_name(subregions)
        if not region_ids:
            ax.set_axis_off()
            return
        self._style_bump_axis(ax, len(region_ids), when_labels)
        self._draw_bump_lines(
            ax, region_ids, rank_map1, rank_map2, id_to_name
        )
