class Where:
    def __init__(self, title: str, region_year: str):
        self.title = title
        self.region_year = region_year

    def get_title(self):
        return self.title

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            + f"({self.get_title()}/{self.region_year})"
        )
