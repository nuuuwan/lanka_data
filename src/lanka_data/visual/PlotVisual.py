import os
from abc import abstractmethod

from lanka_data.visual.plot import Plot
from lanka_data.visual.Visual import Visual


class PlotVisual(Visual):

    @abstractmethod
    def draw(self, dataset, fig):
        pass

    def build(self):
        result = Plot(self).draw()
        image_path = result["image_path"]
        os.system(f"open {image_path}")
        return result
