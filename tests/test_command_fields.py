import pytest

from lanka_data.command.Command import Command
from lanka_data.command.InvalidCommandError import InvalidCommandError
from lanka_data.command.InvalidWhereError import InvalidWhereError
from lanka_data.command.UnknownHowError import UnknownHowError
from lanka_data.command.fields.How import How


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

    def test_where_rejects_traversal_like_token(self):
        with pytest.raises(InvalidWhereError):
            Command.from_str("Religion/2012/LK../Map")

    def test_how_registry_rejects_unknown_modifier(self):
        with pytest.raises(UnknownHowError):
            How("Map:Unknown")
