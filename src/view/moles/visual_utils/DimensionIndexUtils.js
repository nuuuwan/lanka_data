import Region from "../../../nonview/core/thing/concept/category_concept/region/region/Region.js";
export default class DimensionIndexUtils {
  static getDimIndexInfo(datumList) {
    const nDims = datumList[0].query.dimThingList.length;
    const varyingDimIndexes = [];

    for (let dimIndex = 0; dimIndex < nDims; dimIndex++) {
      const values = new Set(
        datumList.map((datum) => datum.query.dimThingList[dimIndex].value),
      );
      if (values.size > 1) {
        varyingDimIndexes.push(dimIndex);
      }
    }

    return { nDims, varyingDimIndexes };
  }

  static isRegionDim(datumList, dimIndex) {
    return datumList[0].query.dimThingList[dimIndex] instanceof Region;
  }

  static getXAxisDimIndex(datumList, stackDimIndex = null) {
    const { varyingDimIndexes } =
      DimensionIndexUtils.getDimIndexInfo(datumList);
    if (varyingDimIndexes.length === 0) {
      return 0;
    }
    if (stackDimIndex === null || varyingDimIndexes.length === 1) {
      return varyingDimIndexes.at(-1);
    }
    return varyingDimIndexes.at(-2);
  }

  static getStackDimIndex(datumList) {
    const { varyingDimIndexes } =
      DimensionIndexUtils.getDimIndexInfo(datumList);
    if (varyingDimIndexes.length < 2) {
      return null;
    }
    return varyingDimIndexes.at(-1);
  }

  static getMarimekkoDimIndexes(datumList) {
    const { varyingDimIndexes } =
      DimensionIndexUtils.getDimIndexInfo(datumList);

    if (varyingDimIndexes.length === 0) {
      return { xAxisDimIndex: 0, stackDimIndex: null };
    }
    if (varyingDimIndexes.length === 1) {
      return { xAxisDimIndex: varyingDimIndexes[0], stackDimIndex: null };
    }

    const regionDimIndex = varyingDimIndexes.find((dimIndex) =>
      DimensionIndexUtils.isRegionDim(datumList, dimIndex),
    );

    if (regionDimIndex !== undefined) {
      const stackDimIndex = varyingDimIndexes.at(-1);
      if (stackDimIndex === regionDimIndex) {
        return {
          xAxisDimIndex: regionDimIndex,
          stackDimIndex: varyingDimIndexes.at(-2),
        };
      }
      return { xAxisDimIndex: regionDimIndex, stackDimIndex };
    }

    return {
      xAxisDimIndex: varyingDimIndexes.at(-2),
      stackDimIndex: varyingDimIndexes.at(-1),
    };
  }

  static getFacetDimIndexes(datumList, xAxisDimIndex, stackDimIndex = null) {
    const { nDims, varyingDimIndexes } =
      DimensionIndexUtils.getDimIndexInfo(datumList);
    return Array.from({ length: nDims }, (_, i) => i).filter(
      (dimIndex) =>
        dimIndex !== xAxisDimIndex &&
        dimIndex !== stackDimIndex &&
        varyingDimIndexes.includes(dimIndex),
    );
  }
}
