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
    { getXLabel, getBarValue, getBarColor, getFacetKey },
  ) {
    const groups = new Map();

    for (const datum of datumList) {
      const facetKey = getFacetKey(datum, facetDimIndexes);
      if (!groups.has(facetKey)) {
        groups.set(facetKey, []);
      }
      groups.get(facetKey).push({
        id: FormatUtils.toTitleCase(getXLabel(datum, xAxisDimIndex)),
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
}
