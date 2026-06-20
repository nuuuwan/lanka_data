from lanka_data.visual.Visual import Visual


class JSONVisual(Visual):
    def build(self):
        return self.datasets[0].get_data_table()
