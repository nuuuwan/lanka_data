from api.command.CommandError import CommandError


class InvalidWhereError(CommandError):
    field = "where"
    code = "invalid_where"
