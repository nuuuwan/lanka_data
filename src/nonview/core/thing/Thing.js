import KeyValue from "../KeyValue.js";
export default class Thing {
  static WILDCARD = "*";
  constructor(value) {
    this.value = value;
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
