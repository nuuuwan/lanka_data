class Where:
    def __init__(self, title: str, region_year: str, description: str):
        self.title = title
        self.region_year = region_year
        self.description = description

    def get_description(self):
        return self.description

    def get_title(self):
        return self.title

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            + f"({self.get_title()}/{self.region_year})"
        )
