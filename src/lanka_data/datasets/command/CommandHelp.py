from lanka_data.api.fields.How import How
from lanka_data.api.fields.When import When
from lanka_data.api.fields.Where import Where
from lanka_data.api.fields.WhatWhenRegistry import WhatWhenRegistry


class CommandHelp:
    SOURCE = "lanka_data"
    SOURCE_URL = "https://github.com/nuuuwan/lanka_data/blob/main/README.md"

    @staticmethod
    def what_to_whens():
        pairs = WhatWhenRegistry.pairs(When.available_values())
        what_to_whens = {}
        for what, when in pairs:
            what_to_whens.setdefault(what, set()).add(when)
        return {
            what: sorted(whens)
            for what, whens in sorted(what_to_whens.items())
        }

    @staticmethod
    def where():
        return dict(
            region_types=Where.available_region_types(),
            operators=Where.available_operators(),
            examples=Where.available_examples(),
        )

    @staticmethod
    def how():
        return dict(
            bases=How.available_bases(),
            modifiers=How.available_modifiers(),
        )

    @staticmethod
    def get_help_result():
        return dict(
            what_to_whens=CommandHelp.what_to_whens(),
            where=CommandHelp.where(),
            how=CommandHelp.how(),
            source=CommandHelp.SOURCE,
            source_url=CommandHelp.SOURCE_URL,
        )
