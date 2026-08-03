import LocationOnIcon from "@mui/icons-material/LocationOn";

import CategoryConcept from "../../CategoryConcept.js";
import RegionDataMixin from "./RegionDataMixin.js";
import RegionMatcherMixin from "./RegionMatcherMixin.js";
import RegionGeoMixin from "./RegionGeoMixin.js";

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
  static getMUIICON() {
    return LocationOnIcon;
  }

  static allowArbitraryValues() {
    return false;
  }
}

applyMixin(Region, RegionDataMixin);
applyMixin(Region, RegionMatcherMixin);
applyMixin(Region, RegionGeoMixin);
