import CategoryConcept from "../../CategoryConcept.js";
import RegionDataMixin from "./RegionDataMixin.js";

export default class Region extends CategoryConcept {
  static allowArbitraryValues() {
    return false;
  }
}

Object.assign(Region, RegionDataMixin);
