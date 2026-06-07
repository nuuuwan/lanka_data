class Where:
    def __init__(self, title):
        self.title = title

    def get_title(self):
        return self.title

    def __str__(self):
        return f"{self.__class__.__name__}({self.get_title()})"
