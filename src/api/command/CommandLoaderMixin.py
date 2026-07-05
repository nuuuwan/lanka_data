from api.utils_future import Log

from api.command.InvalidCommandError import InvalidCommandError

log = Log("CommandLoaderMixin")


class CommandLoaderMixin:
    @classmethod
    def from_str(cls, cmd_str: str):

        if cmd_str == "Help":
            return cls(
                what_cmd=cmd_str, when_cmd="", where_cmd="", how_cmd=""
            )

        tokens = cmd_str.split("/")
        n_tokens = len(tokens)

        if n_tokens != 4:
            raise InvalidCommandError(
                "Invalid command format:"
                + " expected <what>/<when>/<where>/<how>,"
                + f" got '{cmd_str}'",
                cmd_str,
            )

        what_cmd, when_cmd, where_cmd, how_cmd = tokens
        return cls(what_cmd, when_cmd, where_cmd, how_cmd)
