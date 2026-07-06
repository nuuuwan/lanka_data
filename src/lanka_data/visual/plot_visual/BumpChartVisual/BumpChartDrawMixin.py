class BumpChartDrawMixin:
    COLOR_INCREASE = "#d62728"
    COLOR_DECREASE = "#1f77b4"
    COLOR_UNCHANGED = "#c7c7c7"
    LINE_WIDTH = 10
    MARKER_SIZE = 200
    LABEL_FONTSIZE = 12

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
                else (cls.COLOR_DECREASE if delta > 0 else cls.COLOR_UNCHANGED)
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
