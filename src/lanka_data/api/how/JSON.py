from lanka_data.api.how.How import How


class JSON(How):

    def get_inner(self, what, when, where, cmd):
        return self.get_data(what, when, where)
