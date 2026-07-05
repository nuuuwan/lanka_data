from lanka_data.command.CommandError import CommandError


class InvalidWhenError(CommandError):
    field = "when"
    code = "invalid_when"
