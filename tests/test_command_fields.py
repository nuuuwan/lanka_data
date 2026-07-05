import pytest

from lanka_data.api.fields.What import What as APIWhat
from lanka_data.api.fields.When import When as APIWhen
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
        what = What.available_values()[0]
        when = When.available_values()[0]
        where = Where.available_values()[0]
        how = How.available_bases()[0]
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

    def test_where_rejects_traversal_like_token(self):
        with pytest.raises(InvalidWhereError):
            Command.from_str("Religion/2012/LK../Map")

    def test_how_registry_rejects_unknown_modifier(self):
        with pytest.raises(UnknownHowError):
            How("Map:Unknown")

    def test_field_introspection_values_are_constructible(self):
        for field_cls in [What, When, Where, How]:
            for value in field_cls.available_values():
                assert field_cls(value).value == value

    def test_field_introspection_describes_values(self):
        for field_cls in [What, When, Where, How]:
            description = field_cls.describe()
            assert description["name"]
            assert description["values"] == field_cls.available_values()

    def test_command_introspection_describes_api_fields(self):
        description = Command.describe_api()
        assert description["format"] == "<what>/<when>/<where>/<how>"
        assert set(description["fields"]) == {"what", "when", "where", "how"}

    def test_command_introspection_generates_valid_commands(self):
        commands = Command.valid_commands(
            where_values=Where.available_values()[:1],
            how_values=How.available_bases()[:1],
            max_commands=3,
        )
        assert commands
        for command_id in commands:
            assert Command.from_str(command_id).cmd_id == command_id

    def test_api_field_introspection_has_dataset_values(self):
        assert "Religion" in APIWhat.available_groups()["census"]
        assert "2012" in APIWhen.available_values()
        assert "district" in APIWhere.available_region_types()
        assert "LK:district" in APIWhere.available_examples()
