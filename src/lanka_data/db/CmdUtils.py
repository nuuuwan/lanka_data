class CmdUtils:
    @staticmethod
    def get_name_base_from_cmd(cmd: str) -> str:
        file_name_base = cmd.replace("/", "_")
        return file_name_base
