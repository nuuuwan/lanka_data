export default class Thing {
  constructor(value) {
    this.value = value;
  }

  to_kv_pair() {
    return `{${this.constructor.name}}:{${this.value}}`;
  }
}
