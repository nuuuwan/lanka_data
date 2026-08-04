export default class String {
  static SHORTEN_CACHE = new Map();

  static toSnakeCase(value) {
    return `${value}`
      .trim()
      .replace(/&/g, " and ")
      .replace(/[()]/g, "")
      .replace(/([a-z])([A-Z])/g, "$1_$2")
      .replace(/-/g, "_")
      .replace(/\s+/g, "_")
      .replace(/[^a-zA-Z0-9_]+/g, "_")
      .replace(/_+/g, "_")
      .toLowerCase();
  }

  static shorten(value, maxLen) {
    if (maxLen < 0) {
      throw new Error("maxLen must be non-negative");
    }

    const raw = `${value}`;
    const cacheKey = `${raw}::${maxLen}`;
    if (String.SHORTEN_CACHE.has(cacheKey)) {
      return String.SHORTEN_CACHE.get(cacheKey);
    }

    let result;
    if (raw.length <= maxLen) {
      result = raw;
    } else if (maxLen === 0) {
      result = "";
    } else if (maxLen === 1) {
      result = raw[0];
    } else {
      let shortenedLen = maxLen;
      if (shortenedLen > 3) {
        shortenedLen = 3;
      }

      const words = raw.replace(/-/g, " ").split(/\s+/).filter(Boolean);
      if (words.length > 1) {
        result = words
          .map((word) => word[0])
          .join("")
          .toUpperCase();
      } else {
        const chars = [...raw];
        const consonants = chars
          .slice(1)
          .filter((character) => !"aeiou".includes(character.toLowerCase()));
        result = [chars[0], ...consonants.slice(0, shortenedLen - 1)]
          .join("")
          .toUpperCase();
      }
    }

    String.SHORTEN_CACHE.set(cacheKey, result);
    return result;
  }
}
