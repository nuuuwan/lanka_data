import os
import sys

PROJECT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project"
)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
