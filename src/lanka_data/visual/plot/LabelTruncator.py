class LabelTruncator:
    VOWELS = "aeiou"
    TRUNCATE_LEN_FULL = -1
    TRUNCATE_LEN_NONE = 0
    TRUNCATE_LEN_SMALL = 1
    TRUNCATE_LEN_MID = 3
    MAX_REGIONS_FULL_LABEL = 30
    MAX_REGIONS_TRUNCATE_MID = 100
    MAX_REGIONS_TRUNCATE_SMALL = 300

    @classmethod
    def get_truncate_length(cls, region_count):
        thresholds = [
            (cls.MAX_REGIONS_FULL_LABEL, cls.TRUNCATE_LEN_FULL),
            (cls.MAX_REGIONS_TRUNCATE_MID, cls.TRUNCATE_LEN_MID),
            (cls.MAX_REGIONS_TRUNCATE_SMALL, cls.TRUNCATE_LEN_SMALL),
        ]
        for max_count, length in thresholds:
            if region_count <= max_count:
                return length
        return cls.TRUNCATE_LEN_NONE

    @classmethod
    def _truncate(cls, name, n):
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
        if n >= len(consonants) + 1:
            return name[:n].upper()
        return first_char + "".join(consonants[: n - 1]).upper()

    @classmethod
    def get_label(cls, name, region_count):
        n = cls.get_truncate_length(region_count)
        if n == cls.TRUNCATE_LEN_NONE:
            return None
        if n == cls.TRUNCATE_LEN_FULL:
            return name
        return cls._truncate(name, n)
