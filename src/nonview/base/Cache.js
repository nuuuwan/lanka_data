export default class Cache {
  static async get(cacheKey, fallback) {
    if (typeof localStorage !== "undefined") {
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }
    }

    const data = await fallback();
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(cacheKey, JSON.stringify(data));
    }
    return data;
  }
}
