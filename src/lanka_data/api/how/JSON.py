from lanka_data.api.how.How import How


class JSON(How):

    def get_inner(self, command):
        return self.get_data(
            command.get_what(), command.get_when(), command.get_where()
        )
