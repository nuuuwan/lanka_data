from lanka_data.visual.data_export.DataExportVisual import DataExportVisual
from lanka_data.visual.data_export.DelimitedExportMixin import \
    DelimitedExportMixin


class TSVVisual(DelimitedExportMixin, DataExportVisual):
    def build(self):
        return self._render_delimited("\t")
