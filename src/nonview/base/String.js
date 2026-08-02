export default class String {
  static toSnakeCase(value) {
    return String(value)
      .trim()
      .replace(/&/g, " and ")
      .replace(/[()]/g, "")
      .replace(/([a-z])([A-Z])/g, "$1_$2")
      .replace(/\s+/g, "_")
      .replace(/[^a-zA-Z0-9_]+/g, "_")
      .replace(/_+/g, "_")
      .toLowerCase();
  }
}
