class Header:
    TITLE_DELIM = " · "
    TEXT_COLOR = "#000"

    def __init__(self, visual):
        self.visual = visual

    def draw(self, figure_text):
        header_title_items = [
            f"{self.visual.command.what_cmd} ({self.visual.command.when_cmd})",
            self.visual.command.where_cmd,
            self.visual.command.how_cmd,
        ]
        header_title_items = [
            item.strip() for item in header_title_items if item.strip()
        ]
        figure_text(
            (0.5, 0.975),
            self.TITLE_DELIM.join(header_title_items),
            fontsize=16,
            color=self.TEXT_COLOR,
        )
