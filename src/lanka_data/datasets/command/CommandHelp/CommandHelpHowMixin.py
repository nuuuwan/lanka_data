from lanka_data.api.fields.How import How
from lanka_data.api.fields.HowRegistryMixin import HowRegistryMixin


class CommandHelpHowMixin:
    @staticmethod
    def get_how_bases():
        return sorted(HowRegistryMixin.BASE_LABELS.keys())

    @staticmethod
    def get_how_modifiers():
        return sorted(HowRegistryMixin.MODIFIERS.keys())

    @staticmethod
    def _values_for_base(base, modifiers):
        return [base] + [f"{base}:{modifier}" for modifier in modifiers]

    @staticmethod
    def get_how_values():
        bases = CommandHelpHowMixin.get_how_bases()
        modifiers = CommandHelpHowMixin.get_how_modifiers()
        values = []
        for base in bases:
            values += CommandHelpHowMixin._values_for_base(base, modifiers)
        return sorted(values)

    @staticmethod
    def get_how_help():
        values = CommandHelpHowMixin.get_how_values()
        interval_values = sorted(
            value for value in values if How(value).needs_interval
        )
        return {
            "values": values,
            "bases": CommandHelpHowMixin.get_how_bases(),
            "modifiers": CommandHelpHowMixin.get_how_modifiers(),
            "interval_values": interval_values,
        }
