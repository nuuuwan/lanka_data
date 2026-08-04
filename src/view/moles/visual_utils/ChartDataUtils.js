import FormatUtils from "./FormatUtils.js";
import DimensionUtils from "./DimensionUtils.js";
import { getLargestStackKey, sortByStackValue } from "./StackedChartUtils.js";

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

    const facets = Array.from(groups.entries()).map(([facetKey, data]) => ({
      facetKey,
      data: DimensionUtils.sortDataByTime(data, datumList, xAxisDimIndex),
      total: ChartDataUtils.getFacetTotal(data),
    }));
    return DimensionUtils.sortFacets(
      facets,
      datumList,
      facetDimIndexes,
      (a, b) => b.total - a.total,
    );
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

    const allData = Array.from(groups.values()).flatMap((rows) =>
      Array.from(rows.values()),
    );
    const largestStackKey = getLargestStackKey(allData);
    const facets = Array.from(groups.entries()).map(([facetKey, rows]) => {
      const data = sortByStackValue(Array.from(rows.values()), largestStackKey);
      return {
        facetKey,
        data: DimensionUtils.sortDataByTime(data, datumList, xAxisDimIndex),
        total: ChartDataUtils.getFacetTotal(data),
      };
    });
    return DimensionUtils.sortFacets(
      facets,
      datumList,
      facetDimIndexes,
      (a, b) => b.total - a.total,
    );
  }
}
