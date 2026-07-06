import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class ColorSpecConstants:
    DEFAULT_CMAP_ABS = plt.colormaps["YlGn"]
    DEFAULT_CMAP_DIFF = plt.colormaps["coolwarm"]
    DEFAULT_CMAP_CAT = plt.colormaps["rainbow"]

    @staticmethod
    def build_cmap_for_color(rgb):
        light = tuple(1 - (1 - c) * 0.15 for c in rgb[:3])
        return LinearSegmentedColormap.from_list(
            "abs_custom", [light, rgb[:3]]
        )

    # Canonical colors for categorical data fields across all datasets.
    # Labels NOT listed here (ordinal demographic categories and
    # independent/minor party codes like IND01) fall back to the categorical
    # colormap. Colors shared by a religion and its correlated ethnicity
    # (e.g. Hindu/SLTamil) map to a single grouped list.
    COLOR_TO_LABELS = {
        # --- Religion & ethnicity ---
        "#FFBE29": ["Buddhist"],
        "#2000c0": ["OtherChristian"],
        "#c000c0": ["RomanCatholic"],
        "#EB7400": ["Hindu", "SLTamil"],
        "#00534E": ["Islam", "SLMoor"],
        "#8D153A": ["Sinhalese"],
        "#ff2300": ["IndMalaiyagaTamil"],
        "#7570B3": ["Burgher"],
        "#1B9E77": ["Malay"],
        "#E7298A": ["SLChetty"],
        "#A6761D": ["Bharatha"],
        "#66A61E": ["Veddha"],
        # --- Null / placeholder categories ---
        "#222222": ["(Insufficient Data)"],
        "#111111": ["(No Data)"],
        "#dddddd": ["Other"],
        "#cccccc": ["(No Flip)", "(No Segregation)", "(No Change)"],
        # --- Political parties (grouped by alliance / ideology) ---
        # SLFP family (blue)
        "#222288": ["SLFP", "PA", "UPFA"],
        # Muslim & minority parties (dark green)
        "#004400": ["ACMC", "MNA", "NC", "SLMC", "NUA"],
        # UNP family (green)
        "#008800": ["UNP", "NDF", "SJB"],
        "#009900": [],
        # SLPP / Rajapaksa (dark red)
        "#880000": ["SLPP", "OPPP"],
        # SLMP (purple)
        "#880088": ["SLMP"],
        # Independent groups (light grey)
        "#e0e0e0": ["IG", "IG2", "IG3"],
        "#8800ff": ["DUNF"],
        "#0088ff": ["SB"],
        # Left / JVP-NPP (red)
        "#ff0000": [
            "JVP",
            "NMPP",
            "NPP",
            "NPPT",
            "MEP",
            "USA",
            "SLPF",
            "DNA",
            "JJB",
            "LSSP",
            "CP",
            "NSSP",
            "FSP",
            "SEP",
        ],
        # Tamil / Eastern militant-origin parties (orange-red)
        "#ff2200": ["ELMSP", "EPDP", "TMVP", "EROS"],
        # Up-country Tamil (orange)
        "#ff4400": ["CWC", "UPF"],
        # Buddhist nationalist (amber)
        "#ffcc00": ["SU", "JHU"],
        # Tamil nationalist (yellow)
        "#ffdd00": ["AITC", "ITAK", "TULF", "ACTC", "TMK", "TMTK", "IND9"],
        "#ffffff": ["ELJP", "INDI"],
        "#ff8822": ["IND16"],
        # --- Validation only ---
        "#0088f1": ["A"],
        "#ff4401": ["B"],
        # --- Segregation ---
        "#fe0000": ["Segregated"],
        "#ff8800": ["Change"],
        # --- Election summary ---
        "#008801": ["Valid"],
        "#cc0001": ["DidNotVote"],
        "#ff8801": ["Rejected"],
        # Gender
        "#88ccff": ["Male"],
        "#ff88cc": ["Female"],
    }
