from lanka_data.visual.plot.chart.AbstractChart import AbstractChart


class BumpChart(AbstractChart):
    CHART_TYPE = "BumpChart"
    COLOR_INCREASE = "#d62728"
    COLOR_DECREASE = "#1f77b4"
    COLOR_UNCHANGED = "#c7c7c7"
    LINE_WIDTH = 3.6
    MARKER_SIZE = 34
    LABEL_FONTSIZE = 9

    @classmethod
    def _add_segmented_line(cls, ax, rank1, rank2, color):
        start = (0.0, rank1)
        end = (1.0, rank2)

        start_px = ax.transData.transform(start)
        end_px = ax.transData.transform(end)
        dx_total = end_px[0] - start_px[0]
        dy_total = end_px[1] - start_px[1]

        abs_dx_total = abs(dx_total)
        abs_dy_total = abs(dy_total)

        if abs_dy_total < abs_dx_total:
            horizontal_px = (abs_dx_total - abs_dy_total) / 2
            x_direction = 1 if dx_total >= 0 else -1

            elbow1_px = (
                start_px[0] + x_direction * horizontal_px,
                start_px[1],
            )
            elbow2_px = (
                end_px[0] - x_direction * horizontal_px,
                end_px[1],
            )

            inv = ax.transData.inverted()
            elbow1 = tuple(inv.transform(elbow1_px))
            elbow2 = tuple(inv.transform(elbow2_px))
            points = [start, elbow1, elbow2, end]
        else:
            points = [start, end]

        x_vals = [point[0] for point in points]
        y_vals = [point[1] for point in points]
        ax.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=cls.LINE_WIDTH,
            alpha=0.95,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    @staticmethod
    def _has_diff_values(subregions):
        return any(
            subregion.get("values1") is not None
            and subregion.get("values2") is not None
            for subregion in subregions
        )

    @staticmethod
    def _get_region_totals(subregions, value_key):
        region_to_total = {}
        for subregion in subregions:
            region_id = subregion.get("region_id")
            if region_id is None:
                continue
            values = subregion.get(value_key) or {}
            region_to_total[region_id] = sum(values.values())
        return region_to_total

    @staticmethod
    def _get_rank_map(region_to_total):
        ranked_regions = sorted(
            region_to_total.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return {
            region_id: rank
            for rank, (region_id, _) in enumerate(ranked_regions, start=1)
        }

    @classmethod
    def _get_selected_region_ids(cls, rank_map1, rank_map2):
        all_region_ids = set(rank_map1.keys()) | set(rank_map2.keys())
        sorted_region_ids = sorted(
            all_region_ids,
            key=lambda region_id: (
                min(
                    rank_map1.get(region_id, 10**6),
                    rank_map2.get(region_id, 10**6),
                ),
                region_id,
            ),
        )
        return sorted_region_ids

    @staticmethod
    def _get_region_id_to_name(subregions):
        return {
            subregion.get("region_id"): subregion.get("region_name")
            or str(subregion.get("region_id"))
            for subregion in subregions
            if subregion.get("region_id") is not None
        }

    @classmethod
    def _draw_bump_lines(
        cls,
        ax,
        region_ids,
        rank_map1,
        rank_map2,
        region_id_to_name,
    ):
        n_regions = len(region_ids)
        fallback_rank = n_regions + 1
        for region_id in region_ids:
            rank1 = rank_map1.get(region_id, fallback_rank)
            rank2 = rank_map2.get(region_id, fallback_rank)
            rank_delta = rank2 - rank1
            if rank_delta < 0:
                color = cls.COLOR_INCREASE
            elif rank_delta > 0:
                color = cls.COLOR_DECREASE
            else:
                color = cls.COLOR_UNCHANGED
            region_name = region_id_to_name.get(region_id, str(region_id))
            cls._add_segmented_line(ax, rank1, rank2, color)
            ax.scatter(
                [0, 1],
                [rank1, rank2],
                color=color,
                s=cls.MARKER_SIZE,
                zorder=3,
            )

            ax.text(
                -0.03,
                rank1,
                f"{region_name} ({rank1})",
                ha="right",
                va="center",
                fontsize=cls.LABEL_FONTSIZE,
                color=color,
            )
            ax.text(
                1.03,
                rank2,
                f"{region_name} ({rank2})",
                ha="left",
                va="center",
                fontsize=cls.LABEL_FONTSIZE,
                color=color,
            )

    @staticmethod
    def _style_axis(ax, n_regions, when_labels):
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(n_regions + 0.5, 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(when_labels, fontsize=12)
        ax.set_yticks(range(1, n_regions + 1))
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylabel("Region Rank (1 = highest)", fontsize=11)
        ax.grid(axis="y", color="#ddd", linestyle="--", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_axis(self, ax, chart_data):
        subregions = chart_data["subregions"]
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

        region_to_total1 = self._get_region_totals(subregions, "values1")
        region_to_total2 = self._get_region_totals(subregions, "values2")

        rank_map1 = self._get_rank_map(region_to_total1)
        rank_map2 = self._get_rank_map(region_to_total2)
        region_ids = self._get_selected_region_ids(rank_map1, rank_map2)
        region_id_to_name = self._get_region_id_to_name(subregions)

        if not region_ids:
            ax.set_axis_off()
            return

        when_labels = chart_data.get("when_labels") or ["Start", "End"]
        if len(when_labels) != 2:
            when_labels = ["Start", "End"]
        self._style_axis(ax, len(region_ids), when_labels)

        self._draw_bump_lines(
            ax,
            region_ids,
            rank_map1,
            rank_map2,
            region_id_to_name,
        )
