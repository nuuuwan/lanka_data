from lanka_data.visual.VisualFactory import VisualFactory
from lanka_data.visual.HowParam import HowParam


class CommandHelpHowMixin:
    @staticmethod
    def get_how_bases():
        bases = []
        for base_name, visual_cls in VisualFactory._VISUAL_CLS.items():
            if visual_cls is not None:
                bases.append(base_name)
        return sorted(bases)

    @staticmethod
    def get_how_params():
        return {h.label : h.description for h in HowParam.list()}

    @staticmethod
    def get_how_param_descriptions():
        return {
            key: how_param.description
            for key, how_param in HowParam.list().items()
        }

    @staticmethod
    def get_base_label_descriptions():
        descriptions = {}
        for base_name, visual_cls in VisualFactory._VISUAL_CLS.items():
            if visual_cls is not None:
                description = visual_cls.get_description()
                if description is None:
                    description = BASE_LABELS.get(base_name)
                descriptions[base_name] = description
        return descriptions

    @staticmethod
    def get_how_help():
        return {
            "<base>:<Optional param>": {
                "bases": CommandHelpHowMixin.get_how_bases(),
                "params": CommandHelpHowMixin.get_how_params(),
            }
        }
