import { useMemo } from "react";
import { Box, LinearProgress } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import DimensionUtils from "../visual_utils/DimensionUtils.js";
import {
  buildFeatureToDataMap,
  buildGeoVisualData,
  buildRegionLabels,
  getFeatureRegionId,
  getGeoDimInfo,
  getProjectionInfo,
  groupDatumListByFacet,
  matchFeatureToValue,
} from "../visual_utils/GeoVisualUtils.js";
import MultiChartLayout from "../visual_utils/MultiChartLayout.js";
import GeoChoropleth from "./GeoChoropleth.js";
import Legend from "./Legend.js";

export function buildRegionIdToWeight(features, dataMap) {
  const regionIdToWeight = {};
  for (const geoFeature of features) {
    const match = matchFeatureToValue(geoFeature, dataMap);
    if (match) {
      regionIdToWeight[getFeatureRegionId(geoFeature)] = match.items.reduce(
        (total, item) => total + item.value,
        0,
      );
    }
  }
  return regionIdToWeight;
}

export default function Cartogram({ datumSet }) {
  const { datumList } = datumSet;
  const { regionDimIndex, regionClass, stackDimIndex } =
    getGeoDimInfo(datumList);
  const geoJson = useGeoJson(regionClass);

  const { cartograms, legendItems } = useMemo(() => {
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
    const cartograms = groupDatumListByFacet(datumList, facetDimIndexes).map(
      ({ facetKey, facetDatumList }) => {
        const dataMap = buildFeatureToDataMap(
          facetDatumList,
          regionDimIndex,
          stackDimIndex,
        );
        const deformedFeatures = JSON.parse(JSON.stringify(geoFeatures));
        CartogramUtils.compute(
          deformedFeatures,
          buildRegionIdToWeight(geoFeatures, dataMap),
        );
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
          labels: buildRegionLabels(features, projection),
          projectionScale,
          projectionTranslation,
          total: data.reduce((sum, item) => sum + item.value, 0),
        };
      },
    );

    return {
      cartograms: DimensionUtils.sortFacets(
        cartograms,
        datumList,
        facetDimIndexes,
        (a, b) => b.total - a.total,
      ),
      legendItems: Array.from(legendItemMap.values()),
    };
  }, [geoJson, datumList, regionDimIndex, stackDimIndex]);

  if (!geoJson) {
    return <LinearProgress sx={{ m: 2 }} />;
  }

  return (
    <Box data-testid="cartograms">
      {cartograms.length > 1 && (
        <Box data-testid="cartogram-facets" display="none" />
      )}
      <MultiChartLayout
        facets={cartograms.map((cartogram) => ({
          facetKey: cartogram.facetKey,
          data: cartogram,
        }))}
        xAxisDimName={regionClass.name}
        yAxisLabel=""
        renderChart={({ data }) => (
          <GeoChoropleth testId="cartogram" {...data} />
        )}
      />
      <Legend items={legendItems} />
    </Box>
  );
}

Cartogram.IS_CHART = false;
