import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


class FontUtils:
    @staticmethod
    def install_font(font_family: str):
        font_file_path = os.path.join('fonts', font_family + '.ttf')
        if not os.path.exists(font_file_path):
            raise FileNotFoundError(
                f"Font file '{font_file_path}' not found for font family '{font_family}'."
            )
        fm.fontManager.addfont(font_file_path)
        plt.rcParams["font.family"] = font_family
