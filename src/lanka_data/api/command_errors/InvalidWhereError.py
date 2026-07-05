from lanka_data.api.command_errors.CommandError import CommandError


class InvalidWhereError(CommandError):
    field = "where"
    code = "invalid_where"
