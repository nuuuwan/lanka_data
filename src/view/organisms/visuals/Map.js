import { useMemo } from "react";
import { Box } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import LoadingProgress from "../../moles/LoadingProgress.js";
import buildMaps from "../../moles/visual_utils/MapData.js";
import { getGeoDimInfo } from "../../moles/visual_utils/GeoVisualUtils.js";
import GeoChoropleth from "../../moles/visuals/GeoChoropleth.js";
import Legend from "../../moles/visuals/Legend.js";
import MultiChartLayout from "../MultiChartLayout.js";

export default function MapVisual({ datumSet }) {
  const { datumList } = datumSet;
  const { regionDimIndex, regionClass, stackDimIndex } =
    getGeoDimInfo(datumList);
  const geoJson = useGeoJson(regionClass);
  const mapData = useMemo(
    () => buildMaps(geoJson, datumList, regionDimIndex, stackDimIndex),
    [geoJson, datumList, regionDimIndex, stackDimIndex],
  );

  if (!geoJson) {
    return (
      <LoadingProgress ariaLabel="Loading map data" label="Loading map data…" />
    );
  }
  return (
    <Box data-testid="maps">
      {mapData.maps.length > 1 && (
        <Box data-testid="map-facets" display="none" />
      )}
      <MultiChartLayout
        facets={mapData.maps.map((map) => ({
          facetKey: map.facetKey,
          data: map,
        }))}
        xAxisDimName={regionClass.getClassName()}
        yAxisLabel=""
        renderChart={({ data }) => (
          <GeoChoropleth
            testId="map"
            {...data}
            projectionScale={mapData.projectionScale}
            projectionTranslation={mapData.projectionTranslation}
          />
        )}
      />
      <Legend items={mapData.legendItems} />
    </Box>
  );
}

MapVisual.IS_CHART = false;
