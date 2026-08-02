import FormatUtils from "./FormatUtils.js";

export default class ChartDataUtils {
  static getBarValue(datum) {
    const value = parseFloat(datum.answerThing.value);
    return Number.isNaN(value) ? 0 : value;
  }

  static getFacetTotal(data) {
    return data.reduce((sum, item) => sum + item.value, 0);
  }

  static groupDataByFacet(
    datumList,
    xAxisDimIndex,
    facetDimIndexes,
    { getBarValue, getBarColor, getFacetKey },
  ) {
    const groups = new Map();

    for (const datum of datumList) {
      const facetKey = getFacetKey(datum, facetDimIndexes);
      if (!groups.has(facetKey)) {
        groups.set(facetKey, []);
      }
      groups.get(facetKey).push({
        id: FormatUtils.toThingLabel(datum.query.dimThingList[xAxisDimIndex]),
        value: getBarValue(datum),
        color: getBarColor(datum, xAxisDimIndex),
      });
    }

    return Array.from(groups.entries())
      .map(([facetKey, data]) => ({
        facetKey,
        data,
        total: ChartDataUtils.getFacetTotal(data),
      }))
      .sort((a, b) => b.total - a.total);
  }

  static groupStackedDataByFacet(
    datumList,
    xAxisDimIndex,
    stackDimIndex,
    facetDimIndexes,
    { getStackColor, getBarValue, getFacetKey },
  ) {
    const groups = new Map();

    for (const datum of datumList) {
      const facetKey = getFacetKey(datum, facetDimIndexes);
      if (!groups.has(facetKey)) {
        groups.set(facetKey, new Map());
      }
      const facetRows = groups.get(facetKey);
      const xLabel = FormatUtils.toThingLabel(
        datum.query.dimThingList[xAxisDimIndex],
      );
      if (!facetRows.has(xLabel)) {
        facetRows.set(xLabel, { id: xLabel });
      }
      const stackLabel = FormatUtils.toThingLabel(
        datum.query.dimThingList[stackDimIndex],
      );
      facetRows.get(xLabel)[stackLabel] = getBarValue(datum);
      facetRows.get(xLabel)[`${stackLabel}Color`] = getStackColor(
        datum,
        stackDimIndex,
      );
      facetRows.get(xLabel)._barWidth =
        (facetRows.get(xLabel)._barWidth || 0) + getBarValue(datum);
    }

    return Array.from(groups.entries())
      .map(([facetKey, rows]) => {
        const data = Array.from(rows.values());
        return {
          facetKey,
          data,
          total: ChartDataUtils.getFacetTotal(data),
        };
      })
      .sort((a, b) => b.total - a.total);
  }
}
