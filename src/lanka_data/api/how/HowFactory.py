from lanka_data.api.how.cartogram.Cartogram import Cartogram
from lanka_data.api.how.chart.BarChart import BarChart
from lanka_data.api.how.chart.PieChart import PieChart
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

        title_to_how = {
            "JSON": JSON,
            "Map": Map,
            "Cartogram": Cartogram,
            "PieChart": PieChart,
            "BarChart": BarChart,
        }
        how_class = title_to_how.get(title)
        if how_class is not None:
            return how_class(title, params)

        raise ValueError(f"Unknown how title: {title}")
