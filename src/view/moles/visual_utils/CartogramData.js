import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import DimensionUtils from "./DimensionUtils.js";
import {
  buildFeatureToDataMap,
  buildGeoVisualData,
  buildRegionLabels,
  getProjectionInfo,
  groupDatumListByFacet,
  matchFeatureToValue,
} from "./GeoVisualUtils.js";
import {
  buildRegionIdToWeight,
  fitCartogramProjection,
  getGlobalAreaProjectionScales,
} from "./CartogramProjectionUtils.js";

export default function buildCartograms(
  geoJson,
  datumList,
  regionDimIndex,
  stackDimIndex,
) {
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
  const geoFeatures = geoJson.features.filter((geoFeature) =>
    matchFeatureToValue(geoFeature, allDataMap),
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
    const { projection, projectionScale, projectionTranslation } =
      getProjectionInfo(features);
    return {
      facetKey,
      features,
      data,
      projection,
      projectionScale,
      projectionTranslation,
      total: Object.values(regionIdToWeight).reduce(
        (sum, weight) => sum + weight,
        0,
      ),
    };
  });
  const projectionScales = getGlobalAreaProjectionScales(fittedCartograms);
  const cartograms = fittedCartograms.map((cartogram, index) => {
    const { projection, ...fitted } = fitCartogramProjection(
      cartogram,
      projectionScales[index],
    );
    return {
      ...fitted,
      labels: buildRegionLabels(cartogram.features, projection),
    };
  });

  return {
    cartograms: DimensionUtils.sortFacets(
      cartograms,
      datumList,
      facetDimIndexes,
      (a, b) => b.total - a.total,
    ),
    legendItems: Array.from(legendItemMap.values()),
  };
}
