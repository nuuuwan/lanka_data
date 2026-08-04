import DimensionFacetUtils from "./DimensionFacetUtils.js";

export default class DimensionUtils extends DimensionFacetUtils {
  static getDimName(datumList, dimIndex) {
    return datumList[0].query.dimThingList[dimIndex].constructor.getClassName();
  }

  static getXLabel(datum, xAxisDimIndex) {
    const thing = datum.query.dimThingList[xAxisDimIndex];
    return thing.value;
  }

  static getStackLabel(datum, stackDimIndex) {
    const thing = datum.query.dimThingList[stackDimIndex];
    return thing.value;
  }

  static getStackColor(datum, stackDimIndex) {
    return datum.query.dimThingList[stackDimIndex].getColor();
  }

  static getBarColor(datum, xAxisDimIndex) {
    return datum.query.dimThingList[xAxisDimIndex].getColor();
  }
}
