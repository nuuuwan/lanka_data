import matplotlib.pyplot as plt


class ColorSpecConstants:
    DEFAULT_CMAP_ABS = plt.cm.get_cmap("YlGn")
    DEFAULT_CMAP_DIFF = plt.cm.get_cmap("coolwarm")
    DEFAULT_CMAP_CAT = plt.cm.get_cmap("rainbow")

    COLOR_TO_LABELS = {
        # Religion & Ethnicity
        "#FFBE29": ["Buddhist"],
        "#EB7400": ["Hindu", "SLTamil"],
        "#cc8800": ["IndMalaiyagaTamil"],
        "#00534E": ["Islam", "SLMuslim"],
        "#8D153A": ["Sinhalese"],
        "#2000c0": ["OtherChristian"],
        "#c000c0": ["RomanCatholic"],
        # Null
        "#eeeeee": [
            "(No Data)",
            "(Insufficient Data)",
        ],
        "#dddddd": ["Other"],
        "#cccccc": [
            "(No Flip)",
            "(No Segregation)",
            "(No Change)",
        ],
        # Political Parties
        "#222288": ["SLFP", "PA", "UPFA"],
        "#004400": ["ACMC", "MNA", "NC", "SLMC", "NUA"],
        "#008800": ["UNP", "NDF", "SJB"],
        "#009900": [],
        "#880000": ["SLPP", "OPPP"],
        "#880088": ["SLMP"],
        "#e0e0e0": ["IG", "IG2", "IG3"],
        "#8800ff": ["DUNF"],
        "#0088ff": ["SB"],
        "#ff0000": [
            "JVP",
            "NMPP",
            "NPP",
            "MEP",
            "USA",
            "SLPF",
            "DNA",
            "JJB",
            "LSSP",
            "CP",
            "NSSP",
        ],
        "#ff2200": [
            "ELMSP",
            "EPDP",
            "TMVP",
            "EROS",
        ],
        "#ff4400": ["CWC", "UPF"],
        "#ffcc00": ["SU", "JHU"],
        "#ffdd00": ["AITC", "ITAK", "TULF", "ACTC", "IND9"],
        "#ffffff": ["ELJP", "INDI"],
        "#ff8822": ["IND16"],
        # Validation only
        "#0088f1": ["A"],
        "#ff4401": ["B"],
        # Segregation
        "#fe0000": ["Segregated"],
        "#ff8800": ["Change"],
    }
