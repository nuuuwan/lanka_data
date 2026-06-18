from lanka_data.api.how.cartogram.Cartogram import Cartogram
from lanka_data.api.how.JSON import JSON
from lanka_data.api.how.map.Map import Map


class HowFactory:
    @staticmethod
    def from_command(command):
        how_cmd = command.how_cmd
        tokens = how_cmd.split(":")
        title = tokens[0]
        params = None
        if len(tokens) > 1:
            params = tokens[1]

        if title == "JSON":
            return JSON(title, params)

        if title == "Map":
            return Map(title, params)

        if title == "Cartogram":
            return Cartogram(title, params)

        raise ValueError(f"Unknown how title: {title}")
