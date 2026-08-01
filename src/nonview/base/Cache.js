export default class Cache {
  static async get(cacheKey, fallback) {
    const cached = localStorage.getItem(cacheKey);
    if (cached) {
      return JSON.parse(cached);
    }

    const data = await fallback();
    localStorage.setItem(cacheKey, JSON.stringify(data));
    return data;
  }
}
