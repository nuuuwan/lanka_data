import os
import tempfile

DIR_OUTPUT = os.path.join(tempfile.gettempdir(), "lanka_data", "output")


class FileExportMixin:
    def _write_output(self, file_name, data, binary=False):
        output_dir = os.path.join(DIR_OUTPUT, self.command.cmd_id)
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file_name)
        if binary:
            with open(file_path, "wb") as fout:
                fout.write(data)
        else:
            with open(file_path, "w", encoding="utf-8") as fout:
                fout.write(data)
        return {"file_path": file_path}
