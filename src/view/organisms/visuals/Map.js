import { useMemo } from "react";
import { Box } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import LoadingProgress from "../../moles/LoadingProgress.js";
import DimensionUtils from "../../moles/visual_utils/DimensionUtils.js";
import {
  buildFeatureToDataMap,
  buildGeoVisualData,
  buildRegionLabels,
  getGeoDimInfo,
  getProjectionInfo,
  groupDatumListByFacet,
  matchFeatureToValue,
} from "../../moles/visual_utils/GeoVisualUtils.js";
import GeoChoropleth from "../../moles/visuals/GeoChoropleth.js";
import Legend from "../../moles/visuals/Legend.js";
import MultiChartLayout from "../MultiChartLayout.js";

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
