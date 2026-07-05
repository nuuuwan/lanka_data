from lanka_data.api.command_errors.InvalidCommandError import \
    InvalidCommandError
from lanka_data.api.fields import How, What, When, Where


class CommandBaseValidationMixin:
    @staticmethod
    def _validate_cmd_names(cmds):
        known = {"what_cmd", "when_cmd", "where_cmd", "how_cmd"}
        unknown = sorted(set(cmds) - known)
        if unknown:
            raise TypeError(f"Unknown command fields: {', '.join(unknown)}")

    def _validate_parts(self):
        for field_name, value, field_cls in self._field_validation_pairs():
            self._validate_part(field_name, value, field_cls)

    @staticmethod
    def _validate_part(field_name, value, field_cls):
        if isinstance(value, field_cls):
            return
        raise TypeError(
            "Invalid command field type for "
            + f"{field_name}: expected {field_cls.__name__}, "
            + f"got {type(value).__name__}"
        )

    def _field_validation_pairs(self):
        return [
            ("what", self.what, What),
            ("when", self.when, When),
            ("where", self.where, Where),
            ("how", self.how, How),
        ]

    def _validate_coupling(self):
        if self.how.needs_interval and not self.when.is_interval:
            raise InvalidCommandError(
                f"{self.how.value} requires an interval when",
                self.cmd_id,
            )
