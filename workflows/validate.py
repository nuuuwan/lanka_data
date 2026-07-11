import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lanka_data.datasets.WhatLabelValidator import \
    WhatLabelValidator  # noqa: E402

if __name__ == "__main__":
    WhatLabelValidator().validate()
