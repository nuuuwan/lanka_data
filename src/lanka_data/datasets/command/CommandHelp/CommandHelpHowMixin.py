from lanka_data.api.fields.HowRegistryMixin import HowRegistryMixin
from lanka_data.visual.HowParam import HowParam


class CommandHelpHowMixin:
    @staticmethod
    def get_how_bases():
        return sorted(HowRegistryMixin.BASE_LABELS.keys())

    @staticmethod
    def get_how_params():
        return sorted(HowRegistryMixin.MODIFIERS.keys())

    @staticmethod
    def get_how_param_descriptions():
        return {
            key: how_param.description
            for key, how_param in HowParam.list().items()
        }

    @staticmethod
    def get_base_label_descriptions():
        return HowRegistryMixin.BASE_LABELS

    @staticmethod
    def get_how_help():
        return {
            "<base>:<Optional param>": {
                "bases": CommandHelpHowMixin.get_how_bases(),
                "params": CommandHelpHowMixin.get_how_params(),
            }
        }
