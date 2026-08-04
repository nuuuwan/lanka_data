export function deepEqual(left, right) {
  if (left === right) {
    return true;
  }
  if (
    typeof left !== "object" ||
    typeof right !== "object" ||
    left === null ||
    right === null
  ) {
    return false;
  }
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every(
      (key) =>
        Object.prototype.hasOwnProperty.call(right, key) &&
        deepEqual(left[key], right[key]),
    )
  );
}

export function keysOverlap(left, right) {
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index++) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

export function entriesToDeep(entries) {
  const result = {};
  for (const [keys, value] of entries) {
    let node = result;
    for (let index = 0; index < keys.length - 1; index++) {
      const key = keys[index];
      if (!(key in node)) {
        node[key] = {};
      }
      node = node[key];
    }
    node[keys[keys.length - 1]] = value;
  }
  return result;
}

export function deepToEntries(deepObject) {
  const entries = [];
  const visit = (node, path) => {
    if (typeof node !== "object" || node === null || Array.isArray(node)) {
      entries.push([path, node]);
      return;
    }
    for (const [key, child] of Object.entries(node)) {
      visit(child, [...path, key]);
    }
  };
  visit(deepObject, []);
  return entries;
}
