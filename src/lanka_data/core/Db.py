from lanka_data.core.Where import Where


class Db:
    def __init__(self, cmd: str):
        self.cmd = cmd

    def run(self):
        tokens = self.cmd.split("/")
        n_tokens = len(tokens)
        if n_tokens == 1:
            return Where(tokens[0]).run()

        raise ValueError(f"Invalid command: {self.cmd}")
