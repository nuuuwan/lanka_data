import json

from lanka_data.visual.Visual import Visual


class JSONVisual(Visual):
    def build(self):
        print(
            json.dumps(
                self.datasets[0].get_data_table(),
                indent=2,
            )
        )
