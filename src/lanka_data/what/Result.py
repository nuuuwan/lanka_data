class Result:
    def __init__(self, where, what, when):
        self.where = where
        self.what = what
        self.when = when

    def get(self):
        data_list = self.what.get_data_list(self.where)
        source_info = self.what.get_source_info()

        result_data = dict(
            data_list=data_list,
        )
        if self.what.get_values(data_list[0]) is not None:
            result_data["aggr_data"] = self.what.get_aggr_data(data_list)

        result_data = result_data | source_info

        return result_data
