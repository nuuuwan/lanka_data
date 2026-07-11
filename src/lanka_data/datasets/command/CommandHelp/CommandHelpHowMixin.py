from lanka_data.api.fields.How import How
from lanka_data.api.fields.HowRegistryMixin import HowRegistryMixin


class CommandHelpHowMixin:
    @staticmethod
    def get_how_bases():
        return sorted(HowRegistryMixin.BASE_LABELS.keys())

    @staticmethod
    def get_how_params():
        return sorted(HowRegistryMixin.MODIFIERS.keys())


    @staticmethod
    def get_how_help():
        return {
            "<base>:<Optional param>": {
            "bases": CommandHelpHowMixin.get_how_bases(),
            "params": CommandHelpHowMixin.get_how_params(),
            }
        }
