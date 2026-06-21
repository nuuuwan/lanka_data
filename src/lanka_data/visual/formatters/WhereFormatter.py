class WhereFormatter:
    def __init__(self, where_cmd):
        self.where_cmd = where_cmd

    def format(self) -> str:
        return self.where_cmd
