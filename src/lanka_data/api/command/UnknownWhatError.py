from lanka_data.api.command.CommandError import CommandError


class UnknownWhatError(CommandError):
    field = "what"
    code = "unknown_what"
