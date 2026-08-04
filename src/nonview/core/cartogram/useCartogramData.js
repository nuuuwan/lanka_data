import { useMemo } from "react";

import CartogramUtils from "./CartogramUtils.js";
import {
  applyProjectionScale,
  buildRegionIdToWeight,
  getGlobalAreaProjectionScales,
} from "./CartogramDataUtils.js";
import DimensionUtils from "../visual/DimensionUtils.js";
import {
  buildFeatureToDataMap,
  buildGeoVisualData,
  buildRegionLabels,
  getProjectionInfo,
  groupDatumListByFacet,
  matchFeatureToValue,
} from "../visual/GeoVisualUtils.js";

export default function useCartogramData(
  geoJson,
  datumList,
  regionDimIndex,
  stackDimIndex,
) {
  return useMemo(() => {
    if (!geoJson) {
      return { cartograms: [], legendItems: [] };
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
    const legendItemMap = new Map();
    const fittedCartograms = groupDatumListByFacet(
      datumList,
      facetDimIndexes,
    ).map(({ facetKey, facetDatumList }) => {
      const dataMap = buildFeatureToDataMap(
        facetDatumList,
        regionDimIndex,
        stackDimIndex,
      );
      const regionIdToWeight = buildRegionIdToWeight(geoFeatures, dataMap);
      const deformedFeatures = JSON.parse(JSON.stringify(geoFeatures));
      CartogramUtils.compute(deformedFeatures, regionIdToWeight);
      const { features, data } = buildGeoVisualData(
        deformedFeatures,
        dataMap,
        legendItemMap,
      );
      const projectionInfo = getProjectionInfo(features);
      return {
        facetKey,
        features,
        data,
        ...projectionInfo,
        total: Object.values(regionIdToWeight).reduce(
          (sum, weight) => sum + weight,
          0,
        ),
      };
    });
    const projectionScales = getGlobalAreaProjectionScales(fittedCartograms);
    const cartograms = fittedCartograms.map((cartogram, index) => {
      const { projection, ...cartogramData } = cartogram;
      const projectionScale = projectionScales[index];
      const projectionTranslation = applyProjectionScale(
        cartogram,
        projectionScale,
      );
      return {
        ...cartogramData,
        labels: buildRegionLabels(cartogram.features, projection),
        projectionScale,
        projectionTranslation,
      };
    });
    return {
      cartograms: DimensionUtils.sortFacets(
        cartograms,
        datumList,
        facetDimIndexes,
        (left, right) => right.total - left.total,
      ),
      legendItems: Array.from(legendItemMap.values()),
    };
  }, [geoJson, datumList, regionDimIndex, stackDimIndex]);
}
