import { useMemo } from "react";
import { Box } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import LoadingProgress from "../../moles/LoadingProgress.js";
import buildCartograms from "../../moles/visual_utils/CartogramData.js";
import { getGeoDimInfo } from "../../moles/visual_utils/GeoVisualUtils.js";
import GeoChoropleth from "../../moles/visuals/GeoChoropleth.js";
import Legend from "../../moles/visuals/Legend.js";
import MultiChartLayout from "../MultiChartLayout.js";

export {
  buildRegionIdToWeight,
  getGlobalAreaProjectionScales,
  getScaledProjectionTranslation,
} from "../../moles/visual_utils/CartogramProjectionUtils.js";

export default function Cartogram({ datumSet }) {
  const { datumList } = datumSet;
  const { regionDimIndex, regionClass, stackDimIndex } =
    getGeoDimInfo(datumList);
  const geoJson = useGeoJson(regionClass);
  const { cartograms, legendItems } = useMemo(
    () => buildCartograms(geoJson, datumList, regionDimIndex, stackDimIndex),
    [geoJson, datumList, regionDimIndex, stackDimIndex],
  );

  if (!geoJson) {
    return (
      <LoadingProgress ariaLabel="Loading map data" label="Loading map data…" />
    );
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
