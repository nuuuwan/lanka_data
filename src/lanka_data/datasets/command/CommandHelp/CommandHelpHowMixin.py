from lanka_data.visual.HowParam import HowParam
from lanka_data.visual.VisualFactory import VisualFactory


class CommandHelpHowMixin:
    @staticmethod
    def get_how_bases():
        idx = {}
        for visual_cls in VisualFactory.CLS_LIST:
            idx[visual_cls.get_label()] = visual_cls.get_description()
        idx = dict(sorted(idx.items(), key=lambda item: item[0]))
        return idx

    @staticmethod
    def get_how_params():
        return {h.label: h.description for h in HowParam.list()}

    @staticmethod
    def get_how_help():
        return {
            "<base>:<Optional param>": {
                "bases": CommandHelpHowMixin.get_how_bases(),
                "params": CommandHelpHowMixin.get_how_params(),
            }
        }
