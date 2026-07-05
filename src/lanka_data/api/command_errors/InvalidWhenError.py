from lanka_data.api.command_errors.CommandError import CommandError


class InvalidWhenError(CommandError):
    field = "when"
    code = "invalid_when"
