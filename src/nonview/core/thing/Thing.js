export default class Thing {
  constructor(value) {
    this.value = value;
  }

  getHumanReadableValue() {
    return `${this.constructor.name}=${this.value}`;
  }
}
