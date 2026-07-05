from lanka_data.api.command_errors.CommandError import CommandError


class UnknownWhatError(CommandError):
    field = "what"
    code = "unknown_what"
