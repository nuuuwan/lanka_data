import String from "../../../../../../base/String.js";
import WWW from "../../../../../../base/WWW.js";

export default class RegionDataMixin {
  static regionClassId() {
    return this.name.toLowerCase();
  }

  static async loadEnts() {
    const classId = this.regionClassId();
    const url =
      "https://raw.githubusercontent.com" +
      "/nuuuwan/lk_admin_regions/refs/heads/main" +
      `/data/ents/${classId}s.json`;
    return await WWW.json(url);
  }

  // -------------------------------------------------------------------------
  // Functions below need a loaded DataContext
  // -------------------------------------------------------------------------

  static ents = null;

  static assertEntsLoaded() {
    if (!this.ents) {
      throw new Error(
        `Ents not loaded for ${this.name}. ` +
          `Call ${this.name}.loadEnts() before using this method.`,
      );
    }
  }

  static getEntIdxById() {
    this.assertEntsLoaded();
    return Object.fromEntries(this.ents.map((d) => [d.id, d]));
  }

  static getEntIdxByValue() {
    this.assertEntsLoaded();
    return Object.fromEntries(
      this.ents.map((d) => [String.toSnakeCase(d.name), d]),
    );
  }

  static validValues() {
    return Object.keys(this.getEntIdxByValue()).sort();
  }

  getEnt() {
    return this.constructor.getEntIdxByValue()[this.value];
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
}
