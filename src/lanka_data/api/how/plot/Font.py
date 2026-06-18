import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


class Font:
    def __init__(self, font_family: str):
        self.font_family = font_family

    def install(self):
        font_file_path = os.path.join("fonts", self.font_family + ".ttf")
        if not os.path.exists(font_file_path):
            raise FileNotFoundError(
                f"'{font_file_path}' not found for '{self.font_family}'."
            )
        fm.fontManager.addfont(font_file_path)
        plt.rcParams["font.family"] = self.font_family
