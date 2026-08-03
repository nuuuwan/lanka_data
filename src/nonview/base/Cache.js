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
    if (typeof localStorage === "undefined") {
      return data;
    }
    try {
      const payload = JSON.stringify(data);
      const payloadSizeM = (payload.length / (1024 * 1024)).toFixed(2);
      localStorage.setItem(cacheKey, payload);
      if (payloadSizeM > 10) {
        console.error(`🐳 [Cache][ ${payloadSizeM} MB  for "${cacheKey}"`);
      } else if (payloadSizeM > 1) {
        console.warn(`🐘 [Cache][ ${payloadSizeM} MB  for "${cacheKey}"`);
      }
    } catch (e) {
      console.warn(`Failed to cache data for key ${cacheKey}:`, e);
      this.clear();
    }
    return data;
  }
}
