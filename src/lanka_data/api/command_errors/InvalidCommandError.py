from lanka_data.api.command_errors.CommandError import CommandError


class InvalidCommandError(CommandError):
    field = "command"
    code = "invalid_command"
