class Footer:
    def __init__(self, source_list):
        self.source_list = source_list

    def draw(self, figure_text):
        figure_text(
            (0.5, 0.025),
            "Data Sources: " + ", ".join(self.source_list),
            fontsize=16,
            color="#fff",
        )
