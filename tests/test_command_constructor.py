import random

from lanka_data.api.command.Command import Command
from lanka_data.examples.CommandConstructor.CommandConstructor import (
    CommandConstructor,
)


class TestCommandConstructor:
    def test_construct_returns_valid_command_string(self):
        random.seed(0)
        for _ in range(20):
            cmd_str = CommandConstructor.construct()
            assert Command.from_str(cmd_str).cmd_id == cmd_str

    def test_construct_inspects_available_data(self):
        pairs = CommandConstructor.what_when_pairs()
        assert pairs
        assert CommandConstructor.wheres()
        assert CommandConstructor.hows()

    def test_random_cmd_str_has_four_parts(self):
        random.seed(1)
        assert len(CommandConstructor.random_cmd_str().split("/")) == 4

    def test_pairs_exclude_years_without_election_data(self):
        pairs = CommandConstructor.what_when_pairs()
        assert ("Presidential", "2001") not in pairs
        assert ("PresidentialSummary", "2001") not in pairs
        assert ("Presidential", "2015") in pairs
        assert ("PresidentialSummary", "2015") in pairs
