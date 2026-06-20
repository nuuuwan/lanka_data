class Footer:
    def __init__(self, visual):
        self.visual = visual

    def draw(self, figure_text):
        figure_text(
            (0.5, 0.025),
            "Data Sources: " + ", ".join(self.visual.get_source_list()),
            fontsize=16,
            color="#fff",
        )
