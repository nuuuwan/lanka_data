export function deepEqual(a, b) {
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
    (key) =>
      Object.prototype.hasOwnProperty.call(b, key) && deepEqual(a[key], b[key]),
  );
}

export function entriesToDeep(entries) {
  const result = {};
  for (const [keys, value] of entries) {
    let node = result;
    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!(key in node)) node[key] = {};
      node = node[key];
    }
    node[keys.at(-1)] = value;
  }
  return result;
}

export function deepToEntries(deepObject) {
  const entries = [];
  function visit(node, path) {
    if (typeof node !== "object" || node === null || Array.isArray(node)) {
      entries.push([path, node]);
      return;
    }
    for (const [key, child] of Object.entries(node)) {
      visit(child, [...path, key]);
    }
  }
  visit(deepObject, []);
  return entries;
}
