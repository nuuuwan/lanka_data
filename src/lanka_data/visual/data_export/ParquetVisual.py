import io

import pandas as pd

from lanka_data.visual.data_export.DataExportVisual import DataExportVisual
from lanka_data.visual.data_export.FileExportMixin import FileExportMixin


class ParquetVisual(FileExportMixin, DataExportVisual):
    def build(self):
        headers, rows = self.get_table()
        frame = pd.DataFrame(rows, columns=headers)
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        return self._write_output(
            "Data.parquet", buffer.getvalue(), binary=True
        )
