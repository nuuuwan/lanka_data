import District from "./District.js";
import Region from "./region/Region.js";

export default class Province extends Region {
  static getSubRegionClasses() {
    return { District };
  }
}
