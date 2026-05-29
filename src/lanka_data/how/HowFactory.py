from lanka_data.how.JSON import JSON
from lanka_data.how.Map import Map


class HowFactory:
    @staticmethod
    def from_how_cmd(how_cmd: str):
        tokens = how_cmd.split(":")
        title = tokens[0]
        params = None
        if len(tokens) > 1:
            params = tokens[1]

        if title == "JSON":
            return JSON(title, params)

        if title == "Map":
            return Map(title, params)

        raise ValueError(f"Unknown how title: {title}")
