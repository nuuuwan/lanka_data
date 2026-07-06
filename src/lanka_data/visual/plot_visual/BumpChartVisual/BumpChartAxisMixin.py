from lanka_data.visual.plot.Style import Style


class BumpChartAxisMixin:
    @staticmethod
    def _style_bump_axis(ax, n_regions, when_labels):
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(n_regions + 0.5, 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            when_labels,
            fontsize=Style.FONT_SIZE_METADATA,
            color=Style.COLOR_PANEL,
        )
        ax.set_yticks(range(1, n_regions + 1))
        ax.tick_params(
            axis="y",
            labelsize=Style.FONT_SIZE_METADATA,
            labelcolor=Style.COLOR_METADATA,
        )
        ax.set_ylabel(
            "Region Rank (1 = highest)",
            fontsize=Style.FONT_SIZE_METADATA,
            color=Style.COLOR_METADATA,
        )
        ax.grid(
            axis="y", color=Style.COLOR_GRID, linestyle="--", linewidth=0.7
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
