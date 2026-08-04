import { Box } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import { getGeoDimInfo } from "../../../nonview/core/visual/GeoVisualUtils.js";
import useMapData from "../../../nonview/core/visual/useMapData.js";
import LoadingProgress from "../feedback/LoadingProgress.js";
import MultiChartLayout from "../../organisms/MultiChartLayout.js";
import GeoChoropleth from "./GeoChoropleth.js";
import Legend from "./Legend.js";

export default function MapVisual({ datumSet }) {
  const { datumList } = datumSet;
  const { regionDimIndex, regionClass, stackDimIndex } =
    getGeoDimInfo(datumList);
  const geoJson = useGeoJson(regionClass);
  const { maps, legendItems, projectionScale, projectionTranslation } =
    useMapData(geoJson, datumList, regionDimIndex, stackDimIndex);

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
