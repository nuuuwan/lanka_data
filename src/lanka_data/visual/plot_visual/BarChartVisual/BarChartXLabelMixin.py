from lanka_data.visual.plot.Style import Style


class BarChartXLabelMixin:
    _MAX_LABEL_CHARS = 22
    _MIN_FONT = 6
    _LABEL_MARGIN = 0.02
    _MAX_BOTTOM = 0.5

    @classmethod
    def _truncate_label(cls, label):
        label = str(label)
        if len(label) <= cls._MAX_LABEL_CHARS:
            return label
        return label[: cls._MAX_LABEL_CHARS - 1] + "\u2026"

    @staticmethod
    def _max_label_h_frac(ax, fontsize):
        for tick in ax.get_xticklabels():
            tick.set_fontsize(fontsize)
        fig = ax.get_figure()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        subfig_h = fig.bbox.height or 1
        heights = [
            tick.get_window_extent(renderer).height
            for tick in ax.get_xticklabels()
        ]
        return (max(heights) if heights else 0) / subfig_h

    def _fit_x_labels(self, ax, x_labels):
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(
            [self._truncate_label(lbl) for lbl in x_labels],
            rotation=90,
            fontsize=Style.FONT_SIZE_METADATA,
            color=Style.COLOR_METADATA,
        )
        fontsize = Style.FONT_SIZE_METADATA
        h_frac = self._max_label_h_frac(ax, fontsize)
        pos = ax.get_position()
        while (
            pos.y0 + h_frac + self._LABEL_MARGIN > self._MAX_BOTTOM
            and fontsize > self._MIN_FONT
        ):
            fontsize -= 1
            h_frac = self._max_label_h_frac(ax, fontsize)
        return min(h_frac + self._LABEL_MARGIN - pos.y0, self._MAX_BOTTOM)
