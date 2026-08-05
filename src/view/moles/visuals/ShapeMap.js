import { Box, Typography } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import { SHAPE_MAP_OMISSION_FONT_SIZE } from "../../_cons/MapCons.js";
import LoadingProgress from "../../molecules/LoadingProgress.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import { getGeoDimInfo } from "../visual_utils/GeoVisualUtils.js";
import MultiChartLayout from "../../organisms/MultiChartLayout.js";
import Legend from "./Legend.js";
import ShapeMapGraphic from "./ShapeMapGraphic.js";
import ShapeMapScale from "./ShapeMapScale.js";
import useShapeMapData from "./useShapeMapData.js";

export {
  buildShapeMapLayout,
  shareShapeMapScale,
} from "./ShapeMapLayoutUtils.js";

export default function ShapeMap({ datumSet, isUnit = false, shapeConfig }) {
  const { datumList } = datumSet;
  const { regionDimIndex, regionClass, stackDimIndex } =
    getGeoDimInfo(datumList);
  const shapeUnit = `${datumList[0].query.entityClass
    .getClassName()
    .toLowerCase()}s`;
  const geoJson = useGeoJson(regionClass);
  const { maps, legendItems } = useShapeMapData(
    geoJson,
    datumList,
    regionDimIndex,
    stackDimIndex,
    isUnit,
    shapeConfig,
  );
  if (!geoJson) {
    return (
      <LoadingProgress ariaLabel="Loading map data" label="Loading map data…" />
    );
  }
  return (
    <Box data-testid={`${shapeConfig.testId}s`}>
      {maps.length > 1 && (
        <Box data-testid={`${shapeConfig.testId}-facets`} display="none" />
      )}
      <MultiChartLayout
        facets={maps.map((map) => ({ facetKey: map.facetKey, data: map }))}
        xAxisDimName={regionClass.getClassName()}
        yAxisLabel=""
        renderChart={({ data }) => (
          <ShapeMapGraphic data={data} shapeConfig={shapeConfig} />
        )}
      />
      {!isUnit && maps.length > 0 && (
        <>
          <ShapeMapScale
            map={maps[0]}
            shapeName={shapeConfig.shapeName}
            shapeUnit={shapeUnit}
          />
          {maps.some(({ omittedBelow }) => omittedBelow !== null) && (
            <Typography
              variant="caption"
              sx={{
                display: "block",
                fontSize: SHAPE_MAP_OMISSION_FONT_SIZE,
                textAlign: "center",
              }}
            >
              Regions with &lt;
              {FormatUtils.humanizeValue(maps[0].omittedBelow)} {shapeUnit} are
              not displayed
            </Typography>
          )}
        </>
      )}
      <Legend items={legendItems} />
    </Box>
  );
}
