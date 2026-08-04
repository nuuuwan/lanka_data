import Time from "../../../nonview/core/thing/concept/atoms/Time.js";
import DimensionIndexUtils from "./DimensionIndexUtils.js";

export default class DimensionFacetUtils extends DimensionIndexUtils {
  static getFacetKey(datum, facetDimIndexes) {
    return facetDimIndexes
      .map((dimIndex) =>
        datum.query.dimThingList[dimIndex].getHumanReadableValue(),
      )
      .join(" / ");
  }

  static sortFacets(facets, datumList, facetDimIndexes, fallbackCompare) {
    const timeDimIndex = facetDimIndexes.find(
      (dimIndex) => datumList[0].query.dimThingList[dimIndex] instanceof Time,
    );
    if (timeDimIndex === undefined) {
      return facets.sort(fallbackCompare);
    }

    const timeByFacetKey = new Map(
      datumList.map((datum) => [
        DimensionFacetUtils.getFacetKey(datum, facetDimIndexes),
        datum.query.dimThingList[timeDimIndex].value,
      ]),
    );
    return facets.sort((a, b) => {
      const timeComparison = String(
        timeByFacetKey.get(a.facetKey),
      ).localeCompare(String(timeByFacetKey.get(b.facetKey)), undefined, {
        numeric: true,
      });
      return timeComparison || fallbackCompare(a, b);
    });
  }

  static sortDataByTime(data, datumList, dimIndex) {
    if (!(datumList[0].query.dimThingList[dimIndex] instanceof Time)) {
      return data;
    }
    return data.sort((a, b) =>
      String(a.id).localeCompare(String(b.id), undefined, { numeric: true }),
    );
  }
}
