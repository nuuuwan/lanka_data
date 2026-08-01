import Region from "../../../nonview/core/thing/concept/category_concept/region/Region.js";

export default class DimensionUtils {
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
    const { varyingDimIndexes } = DimensionUtils.getDimIndexInfo(datumList);
    if (varyingDimIndexes.length === 0) {
      return 0;
    }
    if (stackDimIndex === null || varyingDimIndexes.length === 1) {
      return varyingDimIndexes.at(-1);
    }
    return varyingDimIndexes.at(-2);
  }

  static getStackDimIndex(datumList) {
    const { varyingDimIndexes } = DimensionUtils.getDimIndexInfo(datumList);
    if (varyingDimIndexes.length < 2) {
      return null;
    }
    return varyingDimIndexes.at(-1);
  }

  static getMarimekkoDimIndexes(datumList) {
    const { varyingDimIndexes } = DimensionUtils.getDimIndexInfo(datumList);

    if (varyingDimIndexes.length === 0) {
      return { xAxisDimIndex: 0, stackDimIndex: null };
    }
    if (varyingDimIndexes.length === 1) {
      return { xAxisDimIndex: varyingDimIndexes[0], stackDimIndex: null };
    }

    const regionDimIndex = varyingDimIndexes.find((dimIndex) =>
      DimensionUtils.isRegionDim(datumList, dimIndex),
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
      DimensionUtils.getDimIndexInfo(datumList);
    return Array.from({ length: nDims }, (_, i) => i).filter(
      (dimIndex) =>
        dimIndex !== xAxisDimIndex &&
        dimIndex !== stackDimIndex &&
        varyingDimIndexes.includes(dimIndex),
    );
  }

  static getFacetKey(datum, facetDimIndexes) {
    return facetDimIndexes
      .map((dimIndex) =>
        datum.query.dimThingList[dimIndex].getHumanReadableValue(),
      )
      .join(" / ");
  }

  static getDimName(datumList, dimIndex) {
    return datumList[0].query.dimThingList[dimIndex].constructor.name;
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
