from lanka_data.visual.Visual import Visual


class JSONVisual(Visual):
    @classmethod
    def get_description(cls):
        return "Exports data as JSON format containing region data with values and categories"

    def build(self):
        return self.datasets[-1].get_data_table()
