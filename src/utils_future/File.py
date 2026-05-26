import os


class File:
    ENCODING = "utf-8"

    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, "r", encoding=self.ENCODING) as f:
            return f.read()

    def write(self, data):
        with open(self.path, "w", encoding=self.ENCODING) as f:
            return f.write(data)

    def exists(self):
        return os.path.exists(self.path)

    def size(self):
        if self.exists():
            return os.path.getsize(self.path)
        return None

    def size_human_readable(self):
        size_bytes = self.size()
        if size_bytes is None:
            return "N/A"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"

    def __str__(self):
        return f"{self.path} ({self.size_human_readable()})"

    def __repr__(self):
        return str(self)
