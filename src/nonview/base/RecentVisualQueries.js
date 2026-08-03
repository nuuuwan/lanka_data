import {
  RECENT_VISUAL_QUERIES_LIMIT,
  RECENT_VISUAL_QUERIES_STORAGE_KEY,
} from "../constants/APP.js";

export default class RecentVisualQueries {
  static getStorage(storage) {
    if (storage) {
      return storage;
    }

    try {
      return typeof localStorage === "undefined" ? null : localStorage;
    } catch {
      return null;
    }
  }

  static read(storage) {
    try {
      const value = this.getStorage(storage)?.getItem(
        RECENT_VISUAL_QUERIES_STORAGE_KEY,
      );
      const queries = value ? JSON.parse(value) : [];
      if (!Array.isArray(queries)) {
        return [];
      }

      return [...new Set(queries.filter((query) => typeof query === "string"))]
        .filter(Boolean)
        .slice(0, RECENT_VISUAL_QUERIES_LIMIT);
    } catch {
      return [];
    }
  }

  static add(query, storage) {
    const queries = [
      query,
      ...this.read(storage).filter((recentQuery) => recentQuery !== query),
    ].slice(0, RECENT_VISUAL_QUERIES_LIMIT);

    try {
      this.getStorage(storage)?.setItem(
        RECENT_VISUAL_QUERIES_STORAGE_KEY,
        JSON.stringify(queries),
      );
    } catch {
      // Recent queries remain available for this page load.
    }

    return queries;
  }

  static clear(storage) {
    try {
      this.getStorage(storage)?.removeItem(RECENT_VISUAL_QUERIES_STORAGE_KEY);
    } catch {
      // Clearing the in-memory menu is still safe when storage is unavailable.
    }

    return [];
  }
}
