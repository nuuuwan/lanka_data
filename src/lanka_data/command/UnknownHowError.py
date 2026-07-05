from lanka_data.command.CommandError import CommandError


class UnknownHowError(CommandError):
    field = "how"
    code = "unknown_how"
