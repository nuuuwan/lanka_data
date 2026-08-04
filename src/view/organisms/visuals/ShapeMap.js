import { Box } from "@mui/material";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import LoadingProgress from "../../moles/LoadingProgress.js";
import { getGeoDimInfo } from "../../moles/visual_utils/GeoVisualUtils.js";
import getShapeMapData from "../../moles/visuals/getShapeMapData.js";
import Legend from "../../moles/visuals/Legend.js";
import ShapeMapGraphic from "../../moles/visuals/ShapeMapGraphic.js";
import ShapeMapScale from "../../moles/visuals/ShapeMapScale.js";
import MultiChartLayout from "../MultiChartLayout.js";

export default function ShapeMap({ datumSet, isUnit = false, shapeConfig }) {
  const { datumList } = datumSet;
  const { regionDimIndex, regionClass, stackDimIndex } =
    getGeoDimInfo(datumList);
  const shapeUnit = `${datumList[0].query.entityClass
    .getClassName()
    .toLowerCase()}s`;
  const geoJson = useGeoJson(regionClass);
  const { maps, legendItems } = getShapeMapData(
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
        <ShapeMapScale
          map={maps[0]}
          shapeName={shapeConfig.shapeName}
          shapeUnit={shapeUnit}
        />
      )}
      <Legend items={legendItems} />
    </Box>
  );
}
