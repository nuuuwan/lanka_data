import { useMemo } from "react";

import DimensionUtils from "./DimensionUtils.js";
import {
  buildFeatureToDataMap,
  buildGeoVisualData,
  buildRegionLabels,
  getProjectionInfo,
  groupDatumListByFacet,
  matchFeatureToValue,
} from "./GeoVisualUtils.js";

export default function useMapData(
  geoJson,
  datumList,
  regionDimIndex,
  stackDimIndex,
) {
  return useMemo(() => {
    if (!geoJson) {
      return {
        maps: [],
        legendItems: [],
        projectionScale: 0,
        projectionTranslation: [0.5, 0.5],
      };
    }
    const facetDimIndexes = DimensionUtils.getFacetDimIndexes(
      datumList,
      regionDimIndex,
      stackDimIndex,
    );
    const allDataMap = buildFeatureToDataMap(
      datumList,
      regionDimIndex,
      stackDimIndex,
    );
    const geoFeatures = geoJson.features.filter((feature) =>
      matchFeatureToValue(feature, allDataMap),
    );
    const { projection, projectionScale, projectionTranslation } =
      getProjectionInfo(geoFeatures);
    const legendItemMap = new Map();
    const maps = groupDatumListByFacet(datumList, facetDimIndexes).map(
      ({ facetKey, facetDatumList }) => {
        const dataMap = buildFeatureToDataMap(
          facetDatumList,
          regionDimIndex,
          stackDimIndex,
        );
        const { features, data } = buildGeoVisualData(
          geoFeatures,
          dataMap,
          legendItemMap,
        );
        return {
          facetKey,
          features,
          data,
          labels: buildRegionLabels(features, projection),
          total: data.reduce((sum, item) => sum + item.value, 0),
        };
      },
    );
    return {
      maps: DimensionUtils.sortFacets(
        maps,
        datumList,
        facetDimIndexes,
        (left, right) => right.total - left.total,
      ),
      legendItems: Array.from(legendItemMap.values()),
      projectionScale,
      projectionTranslation,
    };
  }, [geoJson, datumList, regionDimIndex, stackDimIndex]);
}
