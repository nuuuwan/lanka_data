from dataclasses import dataclass

from utils_future import Log

log = Log("Dataset")


@dataclass
class Dataset:
    what_cmd: str
    when_cmd: str
    where_cmd: str

    def __str__(self):
        return f"Dataset({self.what_cmd}/{self.when_cmd}/{self.where_cmd})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def list_from_command(cls, command):
        what_cmd, when_cmd, where_cmd, _ = command.unpack()
        datasets = []
        if "-" in when_cmd:
            when_cmd_parts = when_cmd.split("-")
            for when_cmd_part in when_cmd_parts:
                datasets.append(cls(what_cmd, when_cmd_part, where_cmd))

        datasets.append(cls(what_cmd, when_cmd, where_cmd))
        log.debug(f"Built {datasets}")
        return datasets
