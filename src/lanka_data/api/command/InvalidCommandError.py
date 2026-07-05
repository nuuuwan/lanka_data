from lanka_data.api.command.CommandError import CommandError


class InvalidCommandError(CommandError):
    field = "command"
    code = "invalid_command"
