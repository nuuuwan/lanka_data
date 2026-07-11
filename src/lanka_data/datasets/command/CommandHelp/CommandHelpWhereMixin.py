

class CommandHelpWhereMixin:
    COMMAND_TO_INFO = {
        "<region_id>": dict(
            description="Returns data for the specified <region_id>.",
            examples=["LK"],
        ),
        "<region_id>:<region_type>": dict(
            description="Returns data for child regions of type <region_type> in <region_id>.",
            examples=["LK:district"],
        ),
        "<region_id1>,<region_id2>": dict(
            description="Returns data for a list of regions.",
            examples=["LK-1,LK-2"],
        ),
        "<region_id1>,<region_id2>,<region_id3>": dict(
            description="Returns data for a list of regions.",
            examples=["LK-1,LK-2,LK-3"],
        ),
        "<region_id1>...<region_id2>": dict(
            description="Returns data for a range of regions.",
            examples=["LK-1...LK-2"],
        ),
        "<region_id>@<distance>": dict(
            description="Returns regions of the same type within a specified distance of <region_id>.",
            examples=["LK-1127025@20"],
        ),
    }

    @staticmethod
    def get_where_help():
        return CommandHelpWhereMixin.COMMAND_TO_INFO
