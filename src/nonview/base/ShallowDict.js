import { deepEqual, deepToEntries, entriesToDeep } from "./ShallowDictUtils.js";
export class KeyError extends Error {
  constructor(key) {
    super(`Key not found: ${JSON.stringify(key)}`);
    this.name = "KeyError";
  }
}

export default class ShallowDict {
  /**
   * @param {Map<Array, any> | Array<[Array, any]>} [initial]
   */
  constructor(initial) {
    this._dict = new Map(); // strKey -> { key, value }
    if (initial) {
      const pairs = initial instanceof Map ? initial.entries() : initial;
      for (const [key, value] of pairs) {
        this._setRaw(key, value);
      }
    }
  }

  _setRaw(key, value) {
    this._dict.set(JSON.stringify(key), { key, value });
  }

  getDict() {
    const obj = {};
    for (const { key, value } of this._dict.values()) {
      obj[JSON.stringify(key)] = value;
    }
    return obj;
  }

  get(key) {
    const entry = this._dict.get(JSON.stringify(key));
    if (!entry) throw new KeyError(key);
    return entry.value;
  }

  set(key, value) {
    for (const [strKey, entry] of Array.from(this._dict.entries())) {
      const existing = entry.key;
      const n = Math.min(key.length, existing.length);
      let matches = true;
      for (let i = 0; i < n; i++) {
        if (key[i] !== existing[i]) {
          matches = false;
          break;
        }
      }
      if (matches) this._dict.delete(strKey);
    }
    this._setRaw(key, value);
  }

  delete(key) {
    if (!this._dict.delete(JSON.stringify(key))) throw new KeyError(key);
  }

  has(key) {
    return this._dict.has(JSON.stringify(key));
  }

  keys() {
    return [...this._dict.values()].map((e) => e.key);
  }

  values() {
    return [...this._dict.values()].map((e) => e.value);
  }

  entries() {
    return [...this._dict.values()].map((e) => [e.key, e.value]);
  }

  get size() {
    return this._dict.size;
  }

  [Symbol.iterator]() {
    return this.keys()[Symbol.iterator]();
  }

  toDeep() {
    return entriesToDeep(this.entries());
  }

  static fromDeep(deepObj) {
    return new ShallowDict(deepToEntries(deepObj));
  }

  equals(other) {
    if (!(other instanceof ShallowDict)) return false;
    return deepEqual(this.toDeep(), other.toDeep());
  }

  add(other) {
    if (!(other instanceof ShallowDict)) {
      throw new TypeError("add() requires another ShallowDict");
    }
    const merged = this.entries();
    for (const [key, value] of other.entries()) merged.push([key, value]);
    return new ShallowDict(merged);
  }
}
