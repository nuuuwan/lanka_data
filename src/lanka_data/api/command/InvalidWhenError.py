from lanka_data.api.command.CommandError import CommandError


class InvalidWhenError(CommandError):
    field = "when"
    code = "invalid_when"
