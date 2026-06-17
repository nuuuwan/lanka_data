class How:
    def __init__(self, how_label: str, params: str):
        self.how_label = how_label
        self.params = params

    def get_description(self):
        description = ""
        param_description = {
            "2nd": "2nd largest category",
            "3rd": "3rd largest category",
            "Bottom": "Smallest category",
            "Top": "Largest category",
        }.get(self.params, self.params)
        if param_description:
            description += param_description
        return description

    def get_title(self):
        return (
            f"{self.how_label} ({self.params})"
            if self.params
            else self.how_label
        )

    def get_data(self, what, when, where):
        data_list = what.get_data_list(where)
        if len(data_list) == 0:
            raise ValueError(
                f"No data found for the specified region: {where}."
            )
        source_info = what.get_source_info()

        result_data = dict(
            data_list=data_list,
        )
        if what.get_values(data_list[0]) is not None:
            result_data["aggr_data"] = what.get_aggregated_value_data(
                data_list
            )

        result_data = result_data | source_info

        return result_data

    def get_inner(self, what, when, where, cmd):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_descriptions(self, what, when, where):
        return dict(
            what_description=what.get_description(),
            when_description=when,
            where_description=where.get_description(),
            how_description=self.get_description(),
        )

    def get_result(self, what, when, where, cmd):
        return (
            self.get_descriptions(what, when, where)
            | self.get_inner(what, when, where, cmd)
            | dict(cmd=cmd)
        )

    def get_descriptions_title(self, what, when, where):
        descriptions = self.get_descriptions(what, when, where)
        what_description = descriptions["what_description"]
        when_description = descriptions["when_description"]
        where_description = descriptions["where_description"]
        how_description = descriptions["how_description"]

        return (
            f"{how_description} of\n{what_description}"
            + f" ({when_description}) for\n{where_description}"
        )

    def without_params(self):
        return How(self.how_label, "")
