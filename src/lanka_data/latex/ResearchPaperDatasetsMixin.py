class ResearchPaperDatasetsMixin:
    DATASETS = [
        (
            "Administrative geography",
            "Current and historical boundary epochs",
            "Department of Census and Statistics",
        ),
        (
            "Census of Population and Housing",
            "2001, 2012, 2024",
            "Department of Census and Statistics",
        ),
        (
            "Elections (Presidential, Parliamentary, Local)",
            "Multiple election years",
            "Election Commission of Sri Lanka",
        ),
        (
            "Rivers (length, catchment)",
            "2026",
            "HydroRIVERS (via lk\\_rivers)",
        ),
    ]

    SOURCES = [
        (
            "Department of Census and Statistics, Sri Lanka",
            "https://www.statistics.gov.lk/",
        ),
        (
            "Census of Population and Housing 2001",
            "https://www.statistics.gov.lk"
            "/Population/StaticalInformation/CPH2001",
        ),
        (
            "Census of Population and Housing 2012",
            "https://www.statistics.gov.lk/Resource/en/Population"
            "/CPH_2011/CPH_2012_5Per_Rpt.pdf",
        ),
        (
            "Census of Population and Housing 2024",
            "https://www.statistics.gov.lk"
            "/Population/StaticalInformation/CPH2024",
        ),
        (
            "Election Commission of Sri Lanka",
            "https://www.elections.gov.lk",
        ),
        (
            "HydroRIVERS (via lk\\_rivers)",
            "https://github.com/nuuuwan/lk_rivers",
        ),
    ]
