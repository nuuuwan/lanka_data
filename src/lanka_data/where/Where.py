class Where:
    def __init__(self, title: str, year: str):
        self.title = title
        self.year = year

    def get_title(self):
        return self.title

    def __str__(self):
        return f"{self.__class__.__name__}({self.get_title()}/{self.year})"
