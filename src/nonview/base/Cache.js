export default class Cache {
  static clear() {
    console.warn("🧹 Clearing cache");
    localStorage.clear();
  }

  static async get(cacheKey, fallback) {
    if (typeof localStorage !== "undefined") {
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }
    }

    const data = await fallback();
    try {
      localStorage.setItem(cacheKey, JSON.stringify(data));
    } catch (e) {
      console.warn(`Failed to cache data for key ${cacheKey}:`, e);
      this.clear();
    }
    return data;
  }
}
