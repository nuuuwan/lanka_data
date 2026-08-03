import { useMemo } from "react";
import { Box, LinearProgress } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import { MAP_HEIGHT, MAP_WIDTH } from "../../_cons/MapCons.js";
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

export function getGlobalAreaProjectionScales(cartograms) {
  const maxTotal = Math.max(...cartograms.map(({ total }) => total), 0);
  if (maxTotal === 0) {
    return cartograms.map(() => 0);
  }

  const areaScaleFactors = cartograms.map(({ total }) =>
    Math.sqrt(total / maxTotal),
  );
  const globalProjectionScale = Math.min(
    ...cartograms
      .map(({ projectionScale }, index) => {
        const areaScaleFactor = areaScaleFactors[index];
        return areaScaleFactor > 0
          ? projectionScale / areaScaleFactor
          : Infinity;
      })
      .filter(Number.isFinite),
  );

  return areaScaleFactors.map(
    (areaScaleFactor) => globalProjectionScale * areaScaleFactor,
  );
}

export function getScaledProjectionTranslation(
  projectionTranslation,
  scaleRatio,
) {
  return projectionTranslation.map(
    (translation) => 0.5 + (translation - 0.5) * scaleRatio,
  );
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
      const { projection, ...cartogramData } = cartogram;
      const projectionScale = projectionScales[index];
      const projectionTranslation = getScaledProjectionTranslation(
        cartogram.projectionTranslation,
        projectionScale / cartogram.projectionScale,
      );
      projection
        .scale(projectionScale)
        .translate([
          projectionTranslation[0] * MAP_WIDTH,
          projectionTranslation[1] * MAP_HEIGHT,
        ]);
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
        xAxisDimName={regionClass.getClassName()}
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
