class Header:
    TITLE_DELIM = " · "

    def __init__(self, command):
        self.command = command

    def draw(self, figure_text):
        header_title_items = [
            f"{self.command.get_what().title} ({self.command.get_when()})",
            self.command.get_where().get_description(),
            self.command.get_how().get_description(),
        ]
        header_title_items = [
            item.strip() for item in header_title_items if item.strip()
        ]
        figure_text(
            (0.5, 0.975),
            self.TITLE_DELIM.join(header_title_items),
            fontsize=16,
            color="#fff",
        )
