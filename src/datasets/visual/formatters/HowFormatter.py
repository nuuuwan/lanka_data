from datasets.command.fields.How import How


class HowFormatter:
    def __init__(self, how_cmd):
        self.how = How(how_cmd)

    def format(self) -> str:
        return self.how.format()
