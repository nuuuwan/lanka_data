from lanka_data.how.How import How


class JSON(How):

    def get_inner(self, where, what, when, cmd):
        return self.get_data(where, what, when)
