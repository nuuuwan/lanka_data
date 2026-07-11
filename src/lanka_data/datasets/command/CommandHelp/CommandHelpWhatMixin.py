from lanka_data.datasets.what_label.WhatLabel import WhatLabel


class CommandHelpWhatMixin:
    @staticmethod
    def get_what_help():
        return {
            group: {w.label: w.description for w in for_group}
            for group, for_group in WhatLabel.list_by_group().items()
        }
