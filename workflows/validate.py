import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lanka_data.datasets.WhatLabelValidator import (  # noqa: E402
    WhatLabelValidator,
)

if __name__ == "__main__":
    WhatLabelValidator().validate()
