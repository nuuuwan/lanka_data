import json

from lanka_data.api.utils_future.File import File


class JSONFile(File):
    ENCODING = "utf-8"

    def read(self):
        with open(self.path, "r", encoding=self.ENCODING) as f:
            return json.load(f)

    def write(self, content):
        with open(self.path, "w", encoding=self.ENCODING) as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
