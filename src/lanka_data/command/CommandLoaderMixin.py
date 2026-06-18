from utils_future import Log

log = Log("CommandLoaderMixin")


class CommandLoaderMixin:
    @classmethod
    def from_str(cls, cmd_str: str):

        if cmd_str == "Help":
            return cls(
                where_cmd=cmd_str, what_cmd="", when_cmd="", how_cmd=""
            )

        tokens = cmd_str.split("/")
        n_tokens = len(tokens)

        if n_tokens != 4:
            raise ValueError(
                "Invalid command format:"
                + " expected <what>/<when>/<where>/<how>,"
                + f" got '{cmd_str}'"
            )

        what_cmd, when_cmd, where_cmd, how_cmd = tokens

        log.debug(f"{what_cmd=}, {when_cmd=}, {where_cmd=}, {how_cmd=}")
        return cls(where_cmd, what_cmd, when_cmd, how_cmd)
