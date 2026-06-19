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

    @staticmethod
    def list_from_command(command):
        what_cmd, when_cmd, where_cmd, _ = command.unpack()
        data_sets = []
        if "-" in when_cmd:
            when_cmd_parts = when_cmd.split("-")
            for when_cmd_part in when_cmd_parts:
                dataset = Dataset(what_cmd, when_cmd_part, where_cmd)
                data_sets.append(dataset)
        data_sets.append(Dataset(what_cmd, when_cmd, where_cmd))
        log.debug(
            f"from {command} c'ed {[str(ds) for ds in data_sets]}",
            data_sets=data_sets,
            command=command,
        )
        return data_sets
