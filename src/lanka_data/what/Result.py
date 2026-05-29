class Result:
    def __init__(self, where, what, when):
        self.where = where
        self.what = what
        self.when = when

    def get(self):
        data_list = self.what.get_data_list(self.where)
        aggr_data = self.what.get_aggr_data(data_list)
        source_info = self.what.get_source_info()

        return (
            dict(
                data_list=data_list,
                aggr_data=aggr_data,
            )
            | source_info
        )
