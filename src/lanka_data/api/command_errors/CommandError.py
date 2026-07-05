class CommandError(ValueError):
    field = "command"
    code = "invalid_command"

    def __init__(self, message, value=None, suggestions=None):
        super().__init__(message)
        self.message = message
        self.value = value
        self.suggestions = suggestions or []

    def to_dict(self):
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "value": self.value,
            "suggestions": self.suggestions,
        }
