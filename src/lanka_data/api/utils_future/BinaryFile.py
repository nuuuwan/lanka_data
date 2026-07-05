from lanka_data.api.utils_future.File import File


class BinaryFile(File):

    def read(self):
        with open(self.path, "rb") as f:
            return f.read()

    def write(self, data):
        with open(self.path, "wb") as f:
            return f.write(data)
