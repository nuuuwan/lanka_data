import Cache from "./Cache.js";

export default class WWW {
  static async jsonNoCache(url) {
    const response = await fetch(url);
    const jsonContent = await response.json();
    const jsonContentSizeM = JSON.stringify(jsonContent).length / (1024 * 1024);
    console.debug(`🌐 ${jsonContentSizeM.toFixed(2)}MB from ${url}`);
    return jsonContent;
  }

  static async json(url) {
    const cacheKey = `cache_${url}`;
    return await Cache.get(cacheKey, async () => {
      const response = await fetch(url);
      return await response.json();
    });
  }
}
