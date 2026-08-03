import { useMemo } from "react";
import { Box } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import LoadingProgress from "../../molecules/LoadingProgress.js";
import DimensionUtils from "../visual_utils/DimensionUtils.js";
import {
  buildFeatureToDataMap,
  buildGeoVisualData,
  buildRegionLabels,
  getGeoDimInfo,
  getProjectionInfo,
  groupDatumListByFacet,
  matchFeatureToValue,
} from "../visual_utils/GeoVisualUtils.js";
import MultiChartLayout from "../../organisms/MultiChartLayout.js";
import GeoChoropleth from "./GeoChoropleth.js";
import Legend from "./Legend.js";

export default function MapVisual({ datumSet }) {
  const { datumList } = datumSet;
  const { regionDimIndex, regionClass, stackDimIndex } =
    getGeoDimInfo(datumList);
  const geoJson = useGeoJson(regionClass);

  const { maps, legendItems, projectionScale, projectionTranslation } =
    useMemo(() => {
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
      const geoFeatures = geoJson.features.filter((geoFeature) =>
        matchFeatureToValue(geoFeature, allDataMap),
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
          (a, b) => b.total - a.total,
        ),
        legendItems: Array.from(legendItemMap.values()),
        projectionScale,
        projectionTranslation,
      };
    }, [geoJson, datumList, regionDimIndex, stackDimIndex]);

  if (!geoJson) {
    return (
      <LoadingProgress ariaLabel="Loading map data" label="Loading map data…" />
    );
  }

  return (
    <Box data-testid="maps">
      {maps.length > 1 && <Box data-testid="map-facets" display="none" />}
      <MultiChartLayout
        facets={maps.map((map) => ({ facetKey: map.facetKey, data: map }))}
        xAxisDimName={regionClass.getClassName()}
        yAxisLabel=""
        renderChart={({ data }) => (
          <GeoChoropleth
            testId="map"
            {...data}
            projectionScale={projectionScale}
            projectionTranslation={projectionTranslation}
          />
        )}
      />
      <Legend items={legendItems} />
    </Box>
  );
}

MapVisual.IS_CHART = false;
