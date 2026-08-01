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

  static getXAxisDimIndex(datumList) {
    const { varyingDimIndexes } = DimensionUtils.getDimIndexInfo(datumList);
    if (varyingDimIndexes.length === 0) {
      return 0;
    }
    return varyingDimIndexes.at(-1);
  }

  static getFacetDimIndexes(datumList) {
    const xAxisDimIndex = DimensionUtils.getXAxisDimIndex(datumList);
    const { nDims, varyingDimIndexes } =
      DimensionUtils.getDimIndexInfo(datumList);
    return Array.from({ length: nDims }, (_, i) => i).filter(
      (dimIndex) =>
        dimIndex !== xAxisDimIndex && varyingDimIndexes.includes(dimIndex),
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

  static getBarColor(datum, xAxisDimIndex) {
    return datum.query.dimThingList[xAxisDimIndex].getColor();
  }
}
