from lanka_data.visual.annotations.Annotations import Annotations
from lanka_data.visual.data_export.DataExportVisual import DataExportVisual
from lanka_data.visual.data_export.FileExportMixin import FileExportMixin


class TableVisual(FileExportMixin, DataExportVisual):
    def build(self):
        return self._write_output("Table.md", self._render_table())

    def _render_table(self):
        headers, rows = self.get_table()
        str_rows = [[str(cell) for cell in row] for row in rows]
        widths = self._column_widths(headers, str_rows)
        lines = [
            self._format_row(headers, widths),
            self._format_separator(widths),
        ]
        for row in str_rows:
            lines.append(self._format_row(row, widths))
        return "\n".join(lines) + self._callout_block()

    def _callout_block(self):
        callouts = Annotations.from_data_table(
            self._get_data_table()
        ).callouts()
        if not callouts:
            return ""
        lines = ["", "", "### What to notice", ""]
        lines += [f"- {item}" for item in callouts]
        return "\n".join(lines)

    @staticmethod
    def _column_widths(headers, rows):
        widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        return widths

    @staticmethod
    def _format_row(cells, widths):
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    @staticmethod
    def _format_separator(widths):
        return "|" + "|".join("-" * (width + 2) for width in widths) + "|"
