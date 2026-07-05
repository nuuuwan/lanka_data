from api.command.CommandError import CommandError


class InvalidWhenError(CommandError):
    field = "when"
    code = "invalid_when"
