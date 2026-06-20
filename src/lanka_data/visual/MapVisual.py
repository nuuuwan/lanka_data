import os

from lanka_data.visual.plot import Plot
from lanka_data.visual.Visual import Visual


class MapVisual(Visual):
    def build(self):
        result = Plot(self).draw()
        image_path = result["image_path"]
        os.system(f"open {image_path}")
        return result
