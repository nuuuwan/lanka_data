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
    # colormap. Labels that share a color (e.g. Hindu/SLTamil) are a religion
    # and its correlated ethnicity.
    LABEL_TO_COLOR = {
        # --- Census: religion & ethnicity ---
        "Buddhist": "#FFBE29",
        "OtherChristian": "#2000c0",
        "RomanCatholic": "#c000c0",
        "Hindu": "#EB7400",
        "SLTamil": "#EB7400",
        "Islam": "#00534E",
        "SLMoor": "#00534E",
        "Sinhalese": "#8D153A",
        "IndMalaiyagaTamil": "#ff2300",
        "Burgher": "#7570B3",
        "Malay": "#1B9E77",
        "SLChetty": "#E7298A",
        "Bharatha": "#A6761D",
        "Veddha": "#66A61E",
        # --- Census: gender ---
        "Male": "#88ccff",
        "Female": "#ff88cd",
        # --- Shared: null / placeholder categories ---
        "(Insufficient Data)": "#222222",
        "(No Data)": "#111111",
        "Other": "#dddddd",
        "(No Flip)": "#cccccc",
        "(No Change)": "#cccccc",
        # --- Election: political parties (by alliance / ideology) ---
        # SLFP family (blue)
        "SLFP": "#222288",
        "PA": "#222288",
        "UPFA": "#222288",
        # Muslim & minority parties (dark green)
        "ACMC": "#004400",
        "MNA": "#004400",
        "NC": "#004400",
        "SLMC": "#004400",
        "NUA": "#004400",
        # UNP family (green)
        "UNP": "#008800",
        "NDF": "#008800",
        "SJB": "#008800",
        # SLPP / Rajapaksa (dark red)
        "SLPP": "#880000",
        "OPPP": "#880000",
        # SLMP (purple)
        "SLMP": "#880088",
        # Independent groups (light grey)
        "IG": "#e0e0e0",
        "IG2": "#e0e0e0",
        "IG3": "#e0e0e0",
        "DUNF": "#8800ff",
        "SB": "#0088ff",
        # Left / JVP-NPP (red)
        "JVP": "#ff0000",
        "NMPP": "#ff0000",
        "NPP": "#ff0000",
        "NPPT": "#ff0000",
        "MEP": "#ff0000",
        "USA": "#ff0000",
        "SLPF": "#ff0000",
        "DNA": "#ff0000",
        "JJB": "#ff0000",
        "LSSP": "#ff0000",
        "CP": "#ff0000",
        "NSSP": "#ff0000",
        "FSP": "#ff0000",
        "SEP": "#ff0000",
        # Tamil / Eastern militant-origin parties (orange-red)
        "ELMSP": "#ff2200",
        "EPDP": "#ff2200",
        "TMVP": "#ff2200",
        "EROS": "#ff2200",
        # Up-country Tamil (orange)
        "CWC": "#ff4400",
        "UPF": "#ff4400",
        # Buddhist nationalist (amber)
        "SU": "#ffcc00",
        "JHU": "#ffcc00",
        # Tamil nationalist (yellow)
        "AITC": "#ffdd00",
        "ITAK": "#ffdd00",
        "TULF": "#ffdd00",
        "ACTC": "#ffdd00",
        "TMK": "#ffdd00",
        "TMTK": "#ffdd00",
        "IND9": "#ffdd00",
        "ELJP": "#ffffff",
        "INDI": "#ffffff",
        "IND16": "#ff8822",
        # --- Election: summary categories ---
        "Valid": "#008801",
        "DidNotVote": "#cc0001",
        "Rejected": "#ff8801",
        # --- Change ---
        "Change": "#ff8800",
        # --- Validation only ---
        "A": "#0088f1",
        "B": "#ff4401",
        #
        "Firewood": "#882200",
        "Gas": "#ffcc00",
    }
