export class KeyError extends Error {
  constructor(key) {
    super(`Key not found: ${JSON.stringify(key)}`);
    this.name = "KeyError";
  }
}

function deepEqual(a, b) {
  if (a === b) return true;
  if (
    typeof a !== "object" ||
    typeof b !== "object" ||
    a === null ||
    b === null
  ) {
    return false;
  }
  const aKeys = Object.keys(a);
  const bKeys = Object.keys(b);
  if (aKeys.length !== bKeys.length) return false;
  return aKeys.every(
    (k) => Object.prototype.hasOwnProperty.call(b, k) && deepEqual(a[k], b[k]),
  );
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
    const result = {};
    for (const [keys, value] of this.entries()) {
      let node = result;
      for (let i = 0; i < keys.length - 1; i++) {
        const k = keys[i];
        if (!(k in node)) node[k] = {};
        node = node[k];
      }
      node[keys[keys.length - 1]] = value;
    }
    return result;
  }

  static fromDeep(deepObj) {
    const flat = [];
    const recurse = (node, path) => {
      if (typeof node !== "object" || node === null || Array.isArray(node)) {
        flat.push([path, node]);
        return;
      }
      for (const [key, child] of Object.entries(node)) {
        recurse(child, [...path, key]);
      }
    };
    recurse(deepObj, []);
    return new ShallowDict(flat);
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
