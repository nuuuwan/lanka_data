from lanka_data.how.How import How


class JSON(How):
    def get_description(self):
        return "JSON data"

    def get_inner(self, where, what, when):
        return self.get_data(where, what, when)
