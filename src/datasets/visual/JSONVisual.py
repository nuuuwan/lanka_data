from datasets.visual.Visual import Visual


class JSONVisual(Visual):
    def build(self):
        return self.datasets[-1].get_data_table()
