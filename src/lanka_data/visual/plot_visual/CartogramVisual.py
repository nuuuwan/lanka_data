from lanka_data.visual.plot_visual.MapVisual import MapVisual


class CartogramVisual(MapVisual):
    @classmethod
    def get_description(cls):
        return (
            'Cartogram: region sizes are distorted to be proportional '
            'to data values, computed using the '
            'Dougenik-Chrisman-Niemeyer algorithm.'
        )
