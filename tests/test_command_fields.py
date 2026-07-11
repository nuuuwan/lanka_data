import pytest

from lanka_data.api.fields.Where import Where as APIWhere
from lanka_data.command.Command import Command
from lanka_data.command.fields.How import How
from lanka_data.command.fields.What import What
from lanka_data.command.fields.When import When
from lanka_data.command.fields.Where import Where
from lanka_data.command.InvalidCommandError import InvalidCommandError
from lanka_data.command.InvalidWhereError import InvalidWhereError
from lanka_data.command.UnknownHowError import UnknownHowError


class TestCommandFields:
    def test_round_trip_parses_four_value_objects(self):
        command = Command.from_str(
            "Religion/2012-2024/LK:district/Map:Change"
        )
        assert command.cmd_id == "Religion/2012-2024/LK:district/Map:Change"
        assert command.what.value == "Religion"
        assert command.when.years == ["2012", "2024"]
        assert command.where.child_region_type == "district"
        assert command.how.modifier == "Change"
        assert Command.from_str("Religion/2024/LK/JSON").how_cmd == "JSON"

    def test_command_accepts_independent_value_objects(self):
        what, when, where, how = "Religion", "2024", "LK", "Map"
        command = Command(What(what), When(when), Where(where), How(how))
        assert command.cmd_id == f"{what}/{when}/{where}/{how}"
        assert command.what_cmd == what
        assert command.when_cmd == when
        assert command.where_cmd == where
        assert command.how_cmd == how

    def test_command_rejects_unknown_fields_before_validation(self):
        with pytest.raises(
            TypeError, match="Unknown command fields: other_cmd"
        ):
            Command(what_cmd="Religion", other_cmd="Other")

    def test_copy_can_clear_empty_string_fields(self):
        command = Command.from_str("Empty/2024/LK/Map")
        copied = command.copy(where_cmd="", how_cmd="")
        assert copied.where_cmd == ""
        assert copied.how_cmd == ""

    def test_change_modifier_requires_interval_when(self):
        with pytest.raises(InvalidCommandError):
            Command.from_str("Religion/2012/LK/Map:Change")

    def test_bump_chart_requires_interval_when(self):
        with pytest.raises(InvalidCommandError):
            Command.from_str("Religion/2012/LK/BumpChart")

    def test_combined_what_exposes_parts(self):
        command = Command.from_str("Religion+Ethnicity/2024/LK:province/Map")
        assert command.what.is_combined
        assert command.what.whats == ["Religion", "Ethnicity"]
        assert command.what_cmd == "Religion+Ethnicity"

    def test_single_what_is_not_combined(self):
        command = Command.from_str("Religion/2024/LK/Map")
        assert not command.what.is_combined
        assert command.what.whats == ["Religion"]

    def test_where_rejects_traversal_like_token(self):
        with pytest.raises(InvalidWhereError):
            Command.from_str("Religion/2012/LK../Map")

    def test_how_registry_rejects_unknown_modifier(self):
        with pytest.raises(UnknownHowError):
            How("BarChart:Unknown")

    def test_map_allows_category_modifier(self):
        how = How("Map:Hindu")
        assert how.base == "Map"
        assert how.modifier == "Hindu"
        assert how.category == "Hindu"


class TestWhereTopPrimitive:
    def test_parses_top_count(self):
        where = APIWhere("LK:rivers#10")
        assert where.top == 10

    def test_no_top_returns_none(self):
        assert APIWhere("LK:rivers").top is None
        assert APIWhere("LK:rivers").region_filter is None

    def test_builds_top_rank_region_filter(self):
        region_filter = APIWhere("LK:rivers#10").region_filter
        assert region_filter.kind == "rank"
        assert region_filter.direction == "Top"
        assert region_filter.count == 10

    def test_strips_top_from_region_parts(self):
        where = APIWhere("LK:rivers#10")
        assert where.parent_part == "LK"
        assert where.child_region_type == "rivers"

    def test_command_preserves_top_in_cmd_id(self):
        command = Command.from_str("Catchment/2026/LK:rivers#10/BarChart")
        assert command.cmd_id == "Catchment/2026/LK:rivers#10/BarChart"
        assert command.where.region_filter.count == 10
