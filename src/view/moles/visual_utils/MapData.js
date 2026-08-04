import DimensionUtils from "./DimensionUtils.js";
import {
  buildFeatureToDataMap,
  buildGeoVisualData,
  buildRegionLabels,
  getProjectionInfo,
  groupDatumListByFacet,
  matchFeatureToValue,
} from "./GeoVisualUtils.js";

export default function buildMaps(
  geoJson,
  datumList,
  regionDimIndex,
  stackDimIndex,
) {
  if (!geoJson) {
    return {
      maps: [],
      legendItems: [],
      projectionScale: 0,
      projectionTranslation: [0.5, 0.5],
    };
  }
  const facetIndexes = DimensionUtils.getFacetDimIndexes(
    datumList,
    regionDimIndex,
    stackDimIndex,
  );
  const allData = buildFeatureToDataMap(
    datumList,
    regionDimIndex,
    stackDimIndex,
  );
  const features = geoJson.features.filter((feature) =>
    matchFeatureToValue(feature, allData),
  );
  const { projection, projectionScale, projectionTranslation } =
    getProjectionInfo(features);
  const legendItemMap = new Map();
  const maps = groupDatumListByFacet(datumList, facetIndexes).map(
    ({ facetKey, facetDatumList }) => {
      const dataMap = buildFeatureToDataMap(
        facetDatumList,
        regionDimIndex,
        stackDimIndex,
      );
      const visualData = buildGeoVisualData(features, dataMap, legendItemMap);
      return {
        facetKey,
        ...visualData,
        labels: buildRegionLabels(visualData.features, projection),
        total: visualData.data.reduce((sum, item) => sum + item.value, 0),
      };
    },
  );
  return {
    maps: DimensionUtils.sortFacets(
      maps,
      datumList,
      facetIndexes,
      (a, b) => b.total - a.total,
    ),
    legendItems: Array.from(legendItemMap.values()),
    projectionScale,
    projectionTranslation,
  };
}
