import * as AxisUtils from "./DimensionAxisUtils.js";
import * as FacetUtils from "./DimensionFacetUtils.js";

export default class DimensionUtils {
  static getDimIndexInfo = AxisUtils.getDimIndexInfo;
  static isRegionDim = AxisUtils.isRegionDim;
  static getXAxisDimIndex = AxisUtils.getXAxisDimIndex;
  static getStackDimIndex = AxisUtils.getStackDimIndex;
  static getMarimekkoDimIndexes = AxisUtils.getMarimekkoDimIndexes;
  static getFacetDimIndexes = FacetUtils.getFacetDimIndexes;
  static getFacetKey = FacetUtils.getFacetKey;
  static sortFacets = FacetUtils.sortFacets;
  static sortDataByTime = FacetUtils.sortDataByTime;
  static getDimName = FacetUtils.getDimName;
  static getXLabel = FacetUtils.getXLabel;
  static getStackLabel = FacetUtils.getStackLabel;
  static getStackColor = FacetUtils.getStackColor;
  static getBarColor = FacetUtils.getBarColor;
}
