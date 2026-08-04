export default class KeyError extends Error {
  constructor(key) {
    super(`Key not found: ${JSON.stringify(key)}`);
    this.name = "KeyError";
  }
}
