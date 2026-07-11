from lanka_data.visual.HOW_PARAMS_DATA import HOW_PARAMS
from lanka_data.visual.VisualFactory import VisualFactory


class CommandHelpHowMixin:
    @staticmethod
    def get_how_bases():
        return sorted(VisualFactory._VISUAL_CLS.keys())

    @staticmethod
    def get_how_params():
        return sorted(HOW_PARAMS.keys())

    @staticmethod
    def get_how_param_descriptions():
        return {
            key: how_param.description
            for key, how_param in HOW_PARAMS.items()
        }

    @staticmethod
    def get_base_label_descriptions():
        return {
            base: visual_cls.get_description()
            for base, visual_cls in VisualFactory._VISUAL_CLS.items()
        }

    @staticmethod
    def get_how_visual_descriptions():
        return {
            visual_cls.__name__: visual_cls.get_description()
            for visual_cls in sorted(
                set(VisualFactory._VISUAL_CLS.values()),
                key=lambda cls: cls.__name__,
            )
        }

    @staticmethod
    def get_how_help():
        return CommandHelpHowMixin.get_how_visual_descriptions()
