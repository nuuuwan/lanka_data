import WWW from "../../../../../../base/WWW.js";

function toSnakeCase(value) {
  return String(value)
    .replace(/&/g, " and ")
    .replace(/[()]/g, "")
    .replace(/([a-z])([A-Z])/g, "$1_$2")
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .toLowerCase();
}

const ENTS_CACHE = new Map();
let initPromise = null;

const REGION_CLASS_IDS = [
  "country",
  "province",
  "district",
  "dsd",
  "ed",
  "gnd",
  "pd",
];

export default class RegionDataMixin {
  getEnt() {
    return this.constructor.getEntIdxByValue()[this.value];
  }

  static regionClassId() {
    return this.name.toLowerCase();
  }

  static getEntsURL() {
    const classId = this.regionClassId();
    return (
      "https://raw.githubusercontent.com" +
      "/nuuuwan/lk_admin_regions/refs/heads/main" +
      `/data/ents/${classId}s.json`
    );
  }

  static getEnts() {
    const classId = this.regionClassId();
    if (!ENTS_CACHE.has(classId)) {
      throw new Error(
        `Region data not loaded. Call await Region.init() before using ${this.name}.`,
      );
    }
    return ENTS_CACHE.get(classId);
  }

  static getEntIdxById() {
    return Object.fromEntries(this.getEnts().map((d) => [d.id, d]));
  }

  static getEntIdxByValue() {
    return Object.fromEntries(
      this.getEnts().map((d) => [toSnakeCase(d.name), d]),
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
      const value = toSnakeCase(idx[regionId].name);
      return new this(value);
    }
    throw new Error(
      `Invalid region_id: ${regionId} for ${this.name}. ` +
        `Valid region_ids: ${Object.keys(idx)}`,
    );
  }

  static async init() {
    if (initPromise) {
      return initPromise;
    }

    initPromise = (async () => {
      await Promise.all(
        REGION_CLASS_IDS.map(async (classId) => {
          if (ENTS_CACHE.has(classId)) {
            return;
          }
          const url =
            "https://raw.githubusercontent.com" +
            "/nuuuwan/lk_admin_regions/refs/heads/main" +
            `/data/ents/${classId}s.json`;
          const dataList = await WWW.json(url);
          ENTS_CACHE.set(classId, dataList);
        }),
      );
    })();

    return initPromise;
  }
  static validValues() {
    return Object.keys(this.getEntIdxByValue()).sort();
  }

  static fromValue(value) {
    const ConceptClass = this;
    if (value === ConceptClass.WILDCARD) {
      return new ConceptClass(value);
    }
    const normalized = toSnakeCase(value);
    const idx = this.getEntIdxByValue();
    if (normalized in idx) {
      return new ConceptClass(normalized);
    }
    if (this.allowArbitraryValues()) {
      return new ConceptClass(normalized);
    }

    console.debug(Object.keys(idx));
    throw new Error(
      `Invalid label2: ${value} for ${this.name}. ` +
        `Valid labels: ${Object.keys(idx)}`,
    );
  }

  static getSubRegionClassByIdKey() {
    const subRegionClassByIdKey = {};
    for (const subRegionClass of Object.values(this.getSubRegionClasses())) {
      subRegionClassByIdKey[subRegionClass.SUB_REGION_ID_KEY] = subRegionClass;
    }
    return subRegionClassByIdKey;
  }

  static getSubRegionClasses() {
    return {};
  }
}
