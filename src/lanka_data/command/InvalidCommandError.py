from lanka_data.command.CommandError import CommandError


class InvalidCommandError(CommandError):
    field = "command"
    code = "invalid_command"
