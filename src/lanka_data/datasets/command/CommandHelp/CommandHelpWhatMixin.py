from lanka_data.datasets.what_label.WhatLabel import WhatLabel


class CommandHelpWhatMixin:
    @staticmethod
    def get_what_groups():
        groups = {
            "special": [],
            "census": [],
            "election": [],
            "election_summary": [],
            "rivers": [],
        }

        for what_label in WhatLabel.list():
            group = what_label.group
            if group in groups:
                groups[group].append(what_label.label)

        for group in groups:
            groups[group] = sorted(groups[group])

        return groups

    @staticmethod
    def get_what_help():
        groups = CommandHelpWhatMixin.get_what_groups()
        values = sorted(
            set(
                label
                for group_labels in groups.values()
                for label in group_labels
            )
        )
        return {
            "values": values,
            "groups": groups,
            "count": len(values),
        }
