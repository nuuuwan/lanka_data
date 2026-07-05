from lanka_data.api.command.CommandError import CommandError


class UnknownHowError(CommandError):
    field = "how"
    code = "unknown_how"
