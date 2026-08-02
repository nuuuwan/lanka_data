import Concept from "../Concept.js";
import Thing from "../../Thing.js";
import String from "../../../../base/String.js";
export default class CategoryConcept extends Concept {
  static validValues() {
    throw new Error(`validValues() not implemented for ${this.name}`);
  }

  static listAll() {
    return this.validValues()
      .map((value) => new this(value))
      .concat([new this(Thing.SPECIAL_VALUE_EXCLUDED_SMALL)]);
  }

  static idx() {
    return Object.fromEntries(this.listAll().map((item) => [item.value, item]));
  }

  static mapAlias() {
    return {};
  }

  static sortedMapAlias() {
    const aliasMap = this.mapAlias();
    const sorted = {};
    for (const key of Object.keys(aliasMap).sort()) {
      sorted[key] = [...new Set(aliasMap[key])].sort();
    }
    return sorted;
  }

  static aliasToValue() {
    const aliasMap = this.sortedMapAlias();
    const idx = {};
    for (const [value, aliases] of Object.entries(aliasMap)) {
      for (const alias of aliases) {
        idx[alias] = value;
      }
    }
    return idx;
  }

  static checkMapAlias() {
    const validValues = new Set(this.validValues());
    for (const validValue of Object.keys(this.mapAlias())) {
      if (!validValues.has(validValue)) {
        throw new Error(
          `Invalid map_alias key: ${validValue} for ${this.name}. ` +
            `Valid values: ${Array.from(validValues)}`,
        );
      }
    }
  }

  static fromValue(value) {
    this.checkMapAlias();

    if (value === Thing.WILDCARD) {
      return new this(value);
    }

    let normalized = `${value}`.replace(/\*/g, "");
    normalized = String.toSnakeCase(normalized);
    normalized = this.aliasToValue()[normalized] ?? normalized;

    const idx = this.idx();
    if (normalized in idx) {
      return idx[normalized];
    }

    if (this.allowArbitraryValues()) {
      return new this(normalized);
    }

    throw new Error(
      `Invalid label: ${value} (normalized: ${normalized}) for ${this.name}. ` +
        `Valid labels: ${Object.keys(idx)}`,
    );
  }

  static allowArbitraryValues() {
    return false;
  }

  static getColorMap() {
    return {};
  }

  getColor() {
    return this.constructor.getColorMap()[this.value];
  }

  static isOrdered() {
    return false;
  }
}
