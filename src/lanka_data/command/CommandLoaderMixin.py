from utils_future import Log

log = Log("CommandLoaderMixin")


class CommandLoaderMixin:
    @classmethod
    def from_str(cls, cmd_str: str):
        tokens = cmd_str.split("/")
        n_tokens = len(tokens)

        if n_tokens != 4:
            raise ValueError(
                "Invalid command format:"
                + " expected <what>/<when>/<where>/<how>,"
                + f" got '{cmd_str}'"
            )

        what_cmd, when_cmd, where_cmd, how_cmd = tokens

        if when_cmd == "2012":
            if "-pre" not in where_cmd:
                tokens = where_cmd.split(":")
                if len(tokens) == 2 and tokens[1] == "dsd":
                    where_cmd = tokens[0] + "-pre2019:" + ":".join(tokens[1:])

        log.debug(f"{what_cmd=}, {when_cmd=}, {where_cmd=}, {how_cmd=}")
        return cls(where_cmd, what_cmd, when_cmd, how_cmd)
