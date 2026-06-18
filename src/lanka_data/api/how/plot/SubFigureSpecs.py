class SubFigureSpecs:
    @staticmethod
    def get(command):
        from lanka_data.command.Command import Command

        when_cmd = command.when_cmd
        if "-" in when_cmd:
            when_parts = when_cmd.split("-")
            command1 = Command(
                command.what_cmd,
                when_parts[0],
                command.where_cmd,
                command.how_cmd,
            )
            command2 = Command(
                command.what_cmd,
                when_parts[1],
                command.where_cmd,
                command.how_cmd,
            )

            spec = {
                when_parts[0]: command1,
                when_parts[1]: command2,
            }

            spec |= {"Change": command}
            return spec

        return {"": command}
