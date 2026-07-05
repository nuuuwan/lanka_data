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
        first = name[0]
        first_lower = first.lower()
        result = [first]
        for ch in name[1:]:
            if len(result) >= n:
                break
            lower = ch.lower()
            if (
                not ch.isalpha()
                or lower in cls.VOWELS
                or lower == first_lower
            ):
                continue
            result.append(ch)
        return "".join(result)

    @classmethod
    def get_label(cls, name, region_count):
        n = cls.get_truncate_length(region_count)
        if n == cls.TRUNCATE_LEN_NONE:
            return None
        if n == cls.TRUNCATE_LEN_FULL:
            return name
        return cls._truncate(name, n)
