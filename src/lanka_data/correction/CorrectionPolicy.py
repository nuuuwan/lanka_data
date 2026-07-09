from dataclasses import dataclass

AUTO = "auto"
WARN = "warn"
RAISE = "raise"


@dataclass(frozen=True)
class CorrectionPolicy:
    lossless: str = AUTO
    lossy: str = AUTO
    destructive: str = RAISE

    def action(self, severity):
        return getattr(self, severity)
