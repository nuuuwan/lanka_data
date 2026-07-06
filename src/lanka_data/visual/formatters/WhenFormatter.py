from lanka_data.api.fields.When import When


class WhenFormatter:
    def __init__(self, when_cmd):
        self.when = When(when_cmd)

    def format(self) -> str:
        if self.when.value == "":
            return None

        if self.when.is_interval:
            return f"{self.when.start}\u2013{self.when.end}"

        return self.when.value
