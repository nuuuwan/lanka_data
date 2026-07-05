from lanka_data.api.command_errors.CommandError import CommandError


class UnknownHowError(CommandError):
    field = "how"
    code = "unknown_how"
