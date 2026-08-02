import KeyValue from "../KeyValue.js";

export default class Thing {
  static WILDCARD = "*";
  static SPECIAL_VALUE_EXCLUDED_SMALL = "excluded_small";

  constructor(value) {
    this.value = String(value);
  }

  static fromValue(value) {
    return new this(value);
  }

  getHumanReadableValue() {
    return `${this.constructor.name}=${this.value}`;
  }

  getColor() {
    return null;
  }

  toKeyValue() {
    if (this.value === Thing.WILDCARD) {
      return this.constructor.name;
    }
    return `${this.constructor.name}${KeyValue.DELIM}${this.value}`;
  }
}
