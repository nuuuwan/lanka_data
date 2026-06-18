from lanka_data.api.what import WhatFactory


class CommandHelp:
    @staticmethod
    def get_help_result():
        return dict(
            what_to_whens=WhatFactory.get_what_to_whens(),
            where=["LK*", "EC-*", "LG-*"],
            how=["JSON", "Map"],
            source="lanka_data",
            source_url="https://github.com"
            + "/nuuuwan/lanka_data/blob/main/README.md",
        )
