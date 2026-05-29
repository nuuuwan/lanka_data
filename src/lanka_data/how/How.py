class How:
    def __init__(self, title: str, params: str):
        self.title = title
        self.params = params

    def get_data(self, where, what, when):
        data_list = what.get_data_list(where)
        source_info = what.get_source_info()

        result_data = dict(
            data_list=data_list,
        )
        if what.get_values(data_list[0]) is not None:
            result_data["aggr_data"] = what.get_aggr_data(data_list)

        result_data = result_data | source_info

        return result_data

    def get_inner(self, where, what, when):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_title_items(self, where, what, when):
        return [
            where.get_title(),
            what.get_title(),
            when,
            self.title,
        ]

    def get_result(self, where, what, when):
        return dict(
            title_items=self.get_title_items(where, what, when),
        ) | self.get_inner(where, what, when)
