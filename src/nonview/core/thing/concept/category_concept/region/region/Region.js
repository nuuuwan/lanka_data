import CategoryConcept from "../../CategoryConcept.js";
import RegionDataMixin from "../RegionDataMixin.js";

function applyMixin(Target, Mixin) {
  for (const name of Object.getOwnPropertyNames(Mixin.prototype)) {
    if (name !== "constructor") {
      Target.prototype[name] = Mixin.prototype[name];
    }
  }
  for (const name of Object.getOwnPropertyNames(Mixin)) {
    if (name !== "prototype" && name !== "name" && name !== "length") {
      Target[name] = Mixin[name];
    }
  }
}

export default class Region extends CategoryConcept {
  static allowArbitraryValues() {
    return false;
  }
}

applyMixin(Region, RegionDataMixin);
