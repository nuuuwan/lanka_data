import String from "../../../../../../base/String.js";
const ENTS_CACHE = new Map();

const REGION_HIERARCHY = {
  District: { parentClassName: "Province", parentIdKey: "province_id" },
  DSD: { parentClassName: "District", parentIdKey: "district_id" },
  ED: { parentClassName: "District", parentIdKey: "district_id" },
};

export default class RegionDataMixin {
  getEnt() {
    return this.constructor.getEntIdxByValue()[this.value];
  }

  static regionClassId() {
    return this.name.toLowerCase();
  }

  static getEnts() {
    const classId = this.regionClassId();
    if (!ENTS_CACHE.has(classId)) {
      throw new Error(
        `Region data not loaded. Use DataProvider before using ${this.name}.`,
      );
    }
    return ENTS_CACHE.get(classId);
  }

  static getEntIdxById() {
    return Object.fromEntries(this.getEnts().map((d) => [d.id, d]));
  }

  static getEntIdxByValue() {
    return Object.fromEntries(
      this.getEnts().map((d) => [String.toSnakeCase(d.name), d]),
    );
  }

  static list() {
    return this.validValues().map((value) => new this(value));
  }

  static idx() {
    return Object.fromEntries(this.list().map((item) => [item.value, item]));
  }

  static fromRegionId(regionId) {
    const idx = this.getEntIdxById();
    if (regionId in idx) {
      const value = String.toSnakeCase(idx[regionId].name);
      return new this(value);
    }
    throw new Error(
      `Invalid region_id: ${regionId} for ${this.name}. ` +
        `Valid region_ids: ${Object.keys(idx)}`,
    );
  }

  static validValues() {
    return Object.keys(this.getEntIdxByValue()).sort();
  }

  static fromValue(value) {
    const ConceptClass = this;
    if (value === ConceptClass.WILDCARD) {
      return new ConceptClass(value);
    }
    const normalized = String.toSnakeCase(value);
    const idx = this.getEntIdxByValue();
    if (normalized in idx) {
      return new ConceptClass(normalized);
    }
    if (this.allowArbitraryValues()) {
      return new ConceptClass(normalized);
    }
    throw new Error(
      `Invalid label: ${value} for ${this.name}. ` +
        `Valid labels: ${Object.keys(idx)}`,
    );
  }

  static getParentRegionInfo() {
    return REGION_HIERARCHY[this.name] || null;
  }

  static load(regionData) {
    ENTS_CACHE.clear();
    for (const [classId, dataList] of Object.entries(regionData)) {
      ENTS_CACHE.set(classId, dataList);
    }
  }
}
