import KeyValue from "../KeyValue.js";

export default class Thing {
  static WILDCARD = "*";
  static SPECIAL_VALUE_EXCLUDED_SMALL = "excluded_small";
  static SPECIAL_VALUE_EXCLUDED_SMALL_COLOR = "#ccc";

  constructor(value) {
    this.value = String(value);
  }

  static fromValue(value) {
    return new this(value);
  }

  getLabel() {
    return this.value;
  }

  getHumanReadableValue() {
    return `${this.constructor.name}=${this.getLabel()}`;
  }

  getColor() {
    if (this.value === Thing.SPECIAL_VALUE_EXCLUDED_SMALL) {
      return Thing.SPECIAL_VALUE_EXCLUDED_SMALL_COLOR;
    }
    return null;
  }

  toKeyValue() {
    if (this.value === Thing.WILDCARD) {
      return this.constructor.name;
    }
    return `${this.constructor.name}${KeyValue.DELIM}${this.value}`;
  }
}
