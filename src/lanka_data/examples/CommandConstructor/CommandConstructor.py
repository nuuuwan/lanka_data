import random

from lanka_data.api.command.Command import Command
from lanka_data.api.fields.How import How
from lanka_data.api.fields.Where import Where


class CommandConstructor:
    MAX_ATTEMPTS = 100

    @classmethod
    def what_when_pairs(cls):
        return Command.valid_what_when_pairs()

    @classmethod
    def wheres(cls):
        return Where.available_examples()

    @classmethod
    def hows(cls):
        return How.available_values()

    @classmethod
    def random_cmd_str(cls):
        what, when = random.choice(cls.what_when_pairs())
        where = random.choice(cls.wheres())
        how = random.choice(cls.hows())
        return f"{what}/{when}/{where}/{how}"

    @classmethod
    def construct(cls):
        for _ in range(cls.MAX_ATTEMPTS):
            cmd_str = cls.random_cmd_str()
            try:
                return Command.from_str(cmd_str).cmd_id
            except ValueError:
                continue
        raise RuntimeError("Could not construct a valid random command")
