from lanka_data.command.fields.How import How
from lanka_data.visual.formatters.HowFormatter import HowFormatter
from lanka_data.visual.plot.color_spec.ColorSpecHelpers import (
    ColorSpecHelpers,
)


class TestHowRegistry:
    def test_rank_modifier_is_single_source_for_labels_and_color_rank(self):
        how = How("Map:2nd")
        assert how.rank == 1
        assert HowFormatter("Map:2nd").format() == "2nd most common"
        assert ColorSpecHelpers.KEY_PARAM_TO_I_RANK["2nd"] == how.rank

    def test_pct_modifier_is_single_source_for_color_pct_rank(self):
        how = How("Map:3rdPct")
        assert how.pct_rank == 2
        assert ColorSpecHelpers.PCT_VALUE_PARAM_TO_KEY["3rdPct"] == 2

    def test_missing_modifier_has_no_modifier_label(self):
        assert How("Map").modifier_label is None

    def test_all_registered_modifiers_compose_with_representative_where(self):
        for modifier in How.MODIFIERS:
            how = How(f"Map:{modifier}")
            assert how.base == "Map"
            assert how.modifier == modifier
