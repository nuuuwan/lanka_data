from lanka_data.visual.Visual import Visual


class JSONVisual(Visual):
    @classmethod
    def get_description(cls):
        return "Exports data as JSON format with region and category values"

    def build(self):
        return self.datasets[-1].get_data_table()
