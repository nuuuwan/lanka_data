import csv
import io


class DelimitedExportMixin:
    def _render_delimited(self, delimiter):
        headers, rows = self.get_table()
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
        return buffer.getvalue()
