from lanka_data.api.command_errors.CommandError import CommandError


class CorrectionLoopError(CommandError):
    field = "command"
    code = "correction_loop"


class DestructiveCorrectionError(CommandError):
    field = "command"
    code = "destructive_correction"


class UnknownMeasurementError(CommandError):
    field = "what"
    code = "unknown_measurement"
