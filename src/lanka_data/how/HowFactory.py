from lanka_data.how.JSON import JSON
from lanka_data.how.Map import Map


class HowFactory:
    @staticmethod
    def from_title(title: str):
        if title == "JSON":
            return JSON(title)

        if title == "Map":
            return Map(title)

        raise ValueError(f"Unknown how title: {title}")
