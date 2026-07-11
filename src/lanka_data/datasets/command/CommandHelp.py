from lanka_data.api.fields.How import How
from lanka_data.api.fields.What import What
from lanka_data.api.fields.When import When
from lanka_data.api.fields.Where import Where


class CommandHelp:
    SOURCE = "lanka_data"
    SOURCE_URL = "https://github.com/nuuuwan/lanka_data/blob/main/README.md"

    FIELDS = dict(What=What, When=When, Where=Where, How=How)

    @staticmethod
    def _describe(field):
        info = dict(field.describe())
        info.pop("name", None)
        return info

    @staticmethod
    def get_help_result():
        result = {
            name: CommandHelp._describe(field)
            for name, field in CommandHelp.FIELDS.items()
        }
        result["source"] = CommandHelp.SOURCE
        result["source_url"] = CommandHelp.SOURCE_URL
        return result
