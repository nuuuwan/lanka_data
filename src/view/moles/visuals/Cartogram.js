import { Box } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import {
  buildRegionIdToWeight,
  getGlobalAreaProjectionScales,
  getScaledProjectionTranslation,
} from "../../../nonview/core/cartogram/CartogramDataUtils.js";
import useCartogramData from "../../../nonview/core/cartogram/useCartogramData.js";
import { getGeoDimInfo } from "../../../nonview/core/visual/GeoVisualUtils.js";
import LoadingProgress from "../feedback/LoadingProgress.js";
import MultiChartLayout from "../../organisms/MultiChartLayout.js";
import GeoChoropleth from "./GeoChoropleth.js";
import Legend from "./Legend.js";

export {
  buildRegionIdToWeight,
  getGlobalAreaProjectionScales,
  getScaledProjectionTranslation,
};

export default function Cartogram({ datumSet }) {
  const { datumList } = datumSet;
  const { regionDimIndex, regionClass, stackDimIndex } =
    getGeoDimInfo(datumList);
  const geoJson = useGeoJson(regionClass);
  const { cartograms, legendItems } = useCartogramData(
    geoJson,
    datumList,
    regionDimIndex,
    stackDimIndex,
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
        facets={cartograms.map((item) => ({
          facetKey: item.facetKey,
          data: item,
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
