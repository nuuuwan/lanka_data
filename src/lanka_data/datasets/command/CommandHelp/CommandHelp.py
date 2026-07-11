from lanka_data.datasets.command.CommandHelp.CommandHelpHowMixin import \
    CommandHelpHowMixin
from lanka_data.datasets.command.CommandHelp.CommandHelpWhatMixin import \
    CommandHelpWhatMixin
from lanka_data.datasets.command.CommandHelp.CommandHelpWhenMixin import \
    CommandHelpWhenMixin
from lanka_data.datasets.command.CommandHelp.CommandHelpWhereMixin import \
    CommandHelpWhereMixin

SOURCE = "lanka_data"
SOURCE_URL = "https://github.com/nuuuwan/lanka_data/blob/main/README.md"


class CommandHelp(
    CommandHelpWhatMixin,
    CommandHelpWhenMixin,
    CommandHelpWhereMixin,
    CommandHelpHowMixin,
):
    @staticmethod
    def get_help_result():
        return {
            "What": CommandHelp.get_what_help(),
            "When": CommandHelp.get_when_help(),
            "Where": CommandHelp.get_where_help(),
            "How": CommandHelp.get_how_help(),
            "source": SOURCE,
            "source_url": SOURCE_URL,
        }
