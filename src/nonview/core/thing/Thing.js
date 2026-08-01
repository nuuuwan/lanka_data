export default class Thing {
  static WILDCARD = "*";
  constructor(value) {
    this.value = value;
  }

  getHumanReadableValue() {
    return `${this.constructor.name}=${this.value}`;
  }
}
