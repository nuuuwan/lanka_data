import Cache from "./Cache.js";

export default class WWW {
  static async jsonNoCache(url) {
    const response = await fetch(url);
    return await response.json();
  }

  static async json(url) {
    const cacheKey = `cache_${url}`;
    return await Cache.get(cacheKey, async () => {
      const response = await fetch(url);
      return await response.json();
    });
  }
}
