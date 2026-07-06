class LabelTruncator:
    VOWELS = "aeiou"
    TRUNCATE_LEN_FULL = -1
    TRUNCATE_LEN_NONE = 0
    TRUNCATE_LEN_SMALL = 1
    TRUNCATE_LEN_MID = 3

    MAX_REGIONS_FULL_LABEL = 10
    MAX_REGIONS_TRUNCATE_MID = 30
    MAX_REGIONS_TRUNCATE_SMALL = 100

    HEX_MAX_REGIONS_FULL_LABEL = 10
    HEX_MAX_REGIONS_TRUNCATE_MID = 30
    HEX_MAX_REGIONS_TRUNCATE_SMALL = 100

    @classmethod
    def _resolve_thresholds(cls, full, mid, small):
        return [
            (full or cls.MAX_REGIONS_FULL_LABEL, cls.TRUNCATE_LEN_FULL),
            (mid or cls.MAX_REGIONS_TRUNCATE_MID, cls.TRUNCATE_LEN_MID),
            (small or cls.MAX_REGIONS_TRUNCATE_SMALL, cls.TRUNCATE_LEN_SMALL),
        ]

    @classmethod
    def get_truncate_length(
        cls,
        region_count,
        max_regions_full_label=None,
        max_regions_truncate_mid=None,
        max_regions_truncate_small=None,
    ):
        thresholds = cls._resolve_thresholds(
            max_regions_full_label,
            max_regions_truncate_mid,
            max_regions_truncate_small,
        )
        for max_count, length in thresholds:
            if region_count <= max_count:
                return length
        return cls.TRUNCATE_LEN_NONE

    @classmethod
    def _truncate(cls, name, n):
        name = name.replace("-", " ")
        words = name.split()
        if len(words) == 1:
            return cls._truncate_single_word(name, n)
        return cls._truncate_single_word("".join([c[0] for c in words]), n)

    @classmethod
    def _truncate_single_word(cls, name, n):
        first_char = name[0]
        consonants = [c for c in name[1:] if c not in cls.VOWELS]
        if n == 1:
            return first_char
        if n > len(consonants) + 1:
            return name[:n].upper()
        return first_char + "".join(consonants[: n - 1]).upper()

    @classmethod
    def get_label(
        cls,
        name,
        region_count,
        max_regions_full_label=None,
        max_regions_truncate_mid=None,
        max_regions_truncate_small=None,
    ):
        n = cls.get_truncate_length(
            region_count,
            max_regions_full_label,
            max_regions_truncate_mid,
            max_regions_truncate_small,
        )
        if n == cls.TRUNCATE_LEN_NONE:
            return None
        if n == cls.TRUNCATE_LEN_FULL:
            return name
        return cls._truncate(name, n)
