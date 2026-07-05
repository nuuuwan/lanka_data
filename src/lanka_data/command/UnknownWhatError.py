from lanka_data.command.CommandError import CommandError


class UnknownWhatError(CommandError):
    field = "what"
    code = "unknown_what"
