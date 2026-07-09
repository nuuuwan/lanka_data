class CorrectionRule:
    name = "rule"
    field = "command"
    severity = "lossless"

    def applies(self, wc):
        raise NotImplementedError

    def apply(self, wc):
        raise NotImplementedError
