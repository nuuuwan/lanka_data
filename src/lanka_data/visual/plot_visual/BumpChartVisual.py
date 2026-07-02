from lanka_data.visual.plot_visual.PlotVisual import PlotVisual


class BumpChartVisual(PlotVisual):
    COLOR_INCREASE = "#d62728"
    COLOR_DECREASE = "#1f77b4"
    COLOR_UNCHANGED = "#c7c7c7"
    LINE_WIDTH = 3.6
    MARKER_SIZE = 34
    LABEL_FONTSIZE = 9

    @classmethod
    def _add_segmented_line(cls, ax, rank1, rank2, color):
        start, end = (0.0, rank1), (1.0, rank2)
        sp = ax.transData.transform(start)
        ep = ax.transData.transform(end)
        dx, dy = ep[0] - sp[0], ep[1] - sp[1]
        if abs(dy) < abs(dx):
            h = (abs(dx) - abs(dy)) / 2
            xd = 1 if dx >= 0 else -1
            inv = ax.transData.inverted()
            e1 = tuple(inv.transform((sp[0] + xd * h, sp[1])))
            e2 = tuple(inv.transform((ep[0] - xd * h, ep[1])))
            points = [start, e1, e2, end]
        else:
            points = [start, end]
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=color,
            linewidth=cls.LINE_WIDTH,
            alpha=0.95,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    @staticmethod
    def _has_diff_values(subregions):
        return any(
            s.get("values1") is not None and s.get("values2") is not None
            for s in subregions
        )

    @staticmethod
    def _get_region_totals(subregions, value_key):
        return {
            s["region_id"]: sum((s.get(value_key) or {}).values())
            for s in subregions
            if s.get("region_id") is not None
        }

    @staticmethod
    def _get_rank_map(region_to_total):
        ranked = sorted(region_to_total.items(), key=lambda x: (-x[1], x[0]))
        return {rid: rank for rank, (rid, _) in enumerate(ranked, start=1)}

    @classmethod
    def _get_selected_region_ids(cls, rank_map1, rank_map2):
        return sorted(
            set(rank_map1) | set(rank_map2),
            key=lambda rid: (
                min(rank_map1.get(rid, 10**6), rank_map2.get(rid, 10**6)),
                rid,
            ),
        )

    @classmethod
    def _draw_bump_lines(
        cls, ax, region_ids, rank_map1, rank_map2, id_to_name
    ):
        fallback = len(region_ids) + 1
        for rid in region_ids:
            r1 = rank_map1.get(rid, fallback)
            r2 = rank_map2.get(rid, fallback)
            delta = r2 - r1
            color = (
                cls.COLOR_INCREASE
                if delta < 0
                else (
                    cls.COLOR_DECREASE if delta > 0 else cls.COLOR_UNCHANGED
                )
            )
            name = id_to_name.get(rid, str(rid))
            cls._add_segmented_line(ax, r1, r2, color)
            ax.scatter(
                [0, 1], [r1, r2], color=color, s=cls.MARKER_SIZE, zorder=3
            )
            ax.text(
                -0.03,
                r1,
                f"{name} ({r1})",
                ha="right",
                va="center",
                fontsize=cls.LABEL_FONTSIZE,
                color=color,
            )
            ax.text(
                1.03,
                r2,
                f"{name} ({r2})",
                ha="left",
                va="center",
                fontsize=cls.LABEL_FONTSIZE,
                color=color,
            )

    @staticmethod
    def _style_bump_axis(ax, n_regions, when_labels):
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

    def _get_rank_maps(self, subregions):
        rank_map1 = self._get_rank_map(
            self._get_region_totals(subregions, "values1")
        )
        rank_map2 = self._get_rank_map(
            self._get_region_totals(subregions, "values2")
        )
        return rank_map1, rank_map2

    @staticmethod
    def _build_id_to_name(subregions):
        return {
            s["region_id"]: s.get("region_name") or str(s["region_id"])
            for s in subregions
            if s.get("region_id") is not None
        }

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
