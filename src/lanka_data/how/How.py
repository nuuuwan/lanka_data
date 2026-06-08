import hashlib


class How:
    def __init__(self, how_label: str, params: str):
        self.how_label = how_label
        self.params = params

    def get_description(self):
        description = self.__class__.__name__
        param_description = {
            "2nd": "2nd largest value",
            "3rd": "3rd largest value",
            "Bottom": "Bottom value",
            "Top": "Top value",
        }.get(self.params, self.params)
        if param_description:
            description += f" ({param_description})"
        return description

    def get_title(self):
        return (
            f"{self.how_label} ({self.params})"
            if self.params
            else self.how_label
        )

    def get_hash(self, where, what, when):
        return hashlib.md5(
            str(self.get_descriptions(where, what, when)).encode()
        ).hexdigest()[:8]

    def get_data(self, where, what, when):
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

    def get_inner(self, where, what, when, cmd):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_descriptions(self, where, what, when):
        return dict(
            what_description=what.get_description(),
            when_description=when,
            where_description=where.get_description(),
            how_description=self.get_description(),
        )

    def get_result(self, where, what, when, cmd):
        return (
            self.get_descriptions(where, what, when)
            | self.get_inner(where, what, when, cmd)
            | dict(cmd=cmd)
        )

    def get_descriptions_title(self, where, what, when):
        descriptions = self.get_descriptions(where, what, when)
        what_description = descriptions["what_description"]
        when_description = descriptions["when_description"]
        where_description = descriptions["where_description"]
        how_description = descriptions["how_description"]

        return (
            f"{how_description} of\n{what_description}"
            + f" ({when_description}) for\n{where_description}"
        )
