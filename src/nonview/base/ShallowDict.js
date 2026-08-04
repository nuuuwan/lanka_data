import KeyError from "./shallow_dict/KeyError.js";
import {
  deepEqual,
  deepToEntries,
  entriesToDeep,
  keysOverlap,
} from "./shallow_dict/ShallowDictUtils.js";

export { KeyError };

export default class ShallowDict {
  constructor(initial) {
    this._dict = new Map();
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
    return Object.fromEntries(
      this.entries().map(([key, value]) => [JSON.stringify(key), value]),
    );
  }

  get(key) {
    const entry = this._dict.get(JSON.stringify(key));
    if (!entry) {
      throw new KeyError(key);
    }
    return entry.value;
  }

  set(key, value) {
    for (const [stringKey, entry] of this._dict.entries()) {
      if (keysOverlap(key, entry.key)) {
        this._dict.delete(stringKey);
      }
    }
    this._setRaw(key, value);
  }

  delete(key) {
    if (!this._dict.delete(JSON.stringify(key))) {
      throw new KeyError(key);
    }
  }

  has(key) {
    return this._dict.has(JSON.stringify(key));
  }

  keys() {
    return [...this._dict.values()].map(({ key }) => key);
  }

  values() {
    return [...this._dict.values()].map(({ value }) => value);
  }

  entries() {
    return [...this._dict.values()].map(({ key, value }) => [key, value]);
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

  static fromDeep(deepObject) {
    return new ShallowDict(deepToEntries(deepObject));
  }

  equals(other) {
    return (
      other instanceof ShallowDict && deepEqual(this.toDeep(), other.toDeep())
    );
  }

  add(other) {
    if (!(other instanceof ShallowDict)) {
      throw new TypeError("add() requires another ShallowDict");
    }
    return new ShallowDict([...this.entries(), ...other.entries()]);
  }
}
