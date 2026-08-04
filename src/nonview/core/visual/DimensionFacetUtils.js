import Time from "../thing/concept/atoms/Time.js";
import { getDimIndexInfo } from "./DimensionAxisUtils.js";

export function getFacetDimIndexes(
  datumList,
  xAxisDimIndex,
  stackDimIndex = null,
) {
  const { nDims, varyingDimIndexes } = getDimIndexInfo(datumList);
  return Array.from({ length: nDims }, (_, index) => index).filter(
    (dimIndex) =>
      dimIndex !== xAxisDimIndex &&
      dimIndex !== stackDimIndex &&
      varyingDimIndexes.includes(dimIndex),
  );
}

export function getFacetKey(datum, facetDimIndexes) {
  return facetDimIndexes
    .map((dimIndex) =>
      datum.query.dimThingList[dimIndex].getHumanReadableValue(),
    )
    .join(" / ");
}

export function sortFacets(
  facets,
  datumList,
  facetDimIndexes,
  fallbackCompare,
) {
  const timeDimIndex = facetDimIndexes.find(
    (dimIndex) => datumList[0].query.dimThingList[dimIndex] instanceof Time,
  );
  if (timeDimIndex === undefined) {
    return facets.sort(fallbackCompare);
  }
  const timeByFacetKey = new Map(
    datumList.map((datum) => [
      getFacetKey(datum, facetDimIndexes),
      datum.query.dimThingList[timeDimIndex].value,
    ]),
  );
  return facets.sort((left, right) => {
    const timeComparison = String(
      timeByFacetKey.get(left.facetKey),
    ).localeCompare(String(timeByFacetKey.get(right.facetKey)), undefined, {
      numeric: true,
    });
    return timeComparison || fallbackCompare(left, right);
  });
}

export function sortDataByTime(data, datumList, dimIndex) {
  if (!(datumList[0].query.dimThingList[dimIndex] instanceof Time)) {
    return data;
  }
  return data.sort((left, right) =>
    String(left.id).localeCompare(String(right.id), undefined, {
      numeric: true,
    }),
  );
}

export const getDimName = (datumList, dimIndex) =>
  datumList[0].query.dimThingList[dimIndex].constructor.getClassName();
export const getXLabel = (datum, dimIndex) =>
  datum.query.dimThingList[dimIndex].value;
export const getStackLabel = getXLabel;
export const getStackColor = (datum, dimIndex) =>
  datum.query.dimThingList[dimIndex].getColor();
export const getBarColor = getStackColor;
