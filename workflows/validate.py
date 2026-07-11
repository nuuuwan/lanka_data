import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lanka_data.datasets.WhatLabelValidator import (  # noqa: E402
    WhatLabelValidator,
)

if __name__ == "__main__":
    validator = WhatLabelValidator()
    is_valid = validator.validate()

    print(validator.get_report())

    if not is_valid:
        sys.exit(1)
