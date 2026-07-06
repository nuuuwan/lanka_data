from lanka_data.visual.data_export.DataExportVisual import DataExportVisual
from lanka_data.visual.data_export.DelimitedExportMixin import (
    DelimitedExportMixin,
)
from lanka_data.visual.data_export.FileExportMixin import FileExportMixin


class TSVVisual(FileExportMixin, DelimitedExportMixin, DataExportVisual):
    def build(self):
        return self._write_output("Data.tsv", self._render_delimited("\t"))
