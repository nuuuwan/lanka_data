from lanka_data.datasets.what_label.WhatLabel import WhatLabel


class CommandHelpWhatMixin:
    @staticmethod
    def get_what_help():
        return {
            w.label :  w.description for w in WhatLabel.list()
        }