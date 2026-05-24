from lanka_data.core.Where import Where


class Db:
    def __init__(self, cmd: str):
        self.cmd = cmd
        tokens = cmd.split("/")
        n_tokens = len(tokens)
        if n_tokens == 1:
            Where(tokens[0])
        else:
            raise ValueError(f"Invalid command: {cmd}")
