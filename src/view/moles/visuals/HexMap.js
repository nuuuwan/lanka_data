import { useMemo } from "react";
import { Box, Typography } from "@mui/material";
import { geoCentroid } from "d3-geo";

import {
  assignShapes,
  buildHexGrid,
  getBestHexLabelFit,
  getHexBoundaryEdges,
  getHexPoints,
  getShapeCounts,
  getValuePerShape,
} from "../../../nonview/base/ShapeMapUtils.js";
import useGeoJson from "../../../nonview/base/useGeoJson.js";
import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import {
  HEX_MAP_EDGE_WIDTH,
  HEX_MAP_MAX_HEXAGONS,
  HEX_MAP_REGION_BORDER_WIDTH,
  HEX_MAP_SCALE_COLOR,
  HEX_MAP_SCALE_FONT_SIZE,
  MAP_BORDER_COLOR,
  MAP_HEIGHT,
  MAP_LABEL_DARK_COLOR,
  MAP_LABEL_LIGHT_COLOR,
  MAP_PADDING,
  MAP_UNKNOWN_COLOR,
  MAP_WIDTH,
} from "../../_cons/MapCons.js";
import DimensionUtils from "../visual_utils/DimensionUtils.js";
import {
  buildFeatureToDataMap,
  getFeatureRegionId,
  getFittedLabelFontSize,
  getGeoDimInfo,
  groupDatumListByFacet,
  getProjectionInfo,
  matchFeatureToValue,
} from "../visual_utils/GeoVisualUtils.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import MultiChartLayout from "../visual_utils/MultiChartLayout.js";
import Legend from "./Legend.js";

function getDisplayItem(items) {
  return items.reduce((best, item) => (item.value > best.value ? item : best));
}

function buildFacetInfo(geoFeatures, facetKey, dataMap) {
  const regions = geoFeatures
    .map((feature) => {
      const match = matchFeatureToValue(feature, dataMap);
      if (!match) {
        return null;
      }
      return {
        display: getDisplayItem(match.items),
        feature,
        id: String(getFeatureRegionId(feature)),
        weight: match.items.reduce((total, item) => total + item.value, 0),
      };
    })
    .filter(Boolean);
  return { facetKey, regions };
}

function getLabels(shapes, regionById, radius) {
  const centersById = new Map();
  for (const { id, center } of shapes) {
    const centers = centersById.get(id) ?? [];
    centers.push(center);
    centersById.set(id, centers);
  }
  return [...centersById].map(([id, centers]) => ({
    ...getBestHexLabelFit(centers, radius),
    color: regionById.get(id).display.color,
    id,
    name: regionById.get(id).feature.properties.name,
  })).map((label) => ({
    ...label,
    fontSize: getFittedLabelFontSize(label.name, label.width, label.height),
  }));
}

export function buildHexMapLayout(facetInfo, valuePerHexagon) {
  const warpedFeatures = facetInfo.regions.map(({ feature }) =>
    JSON.parse(JSON.stringify(feature)),
  );
  const regionIdToWeight = Object.fromEntries(
    facetInfo.regions.map(({ id, weight }) => [id, weight]),
  );
  if (Object.values(regionIdToWeight).some((weight) => weight > 0)) {
    CartogramUtils.compute(warpedFeatures, regionIdToWeight);
  }
  const { projection } = getProjectionInfo(warpedFeatures);
  const counts = getShapeCounts(regionIdToWeight, valuePerHexagon);
  const totalCount = Object.values(counts).reduce(
    (total, count) => total + count,
    0,
  );
  const { centers, radius } = buildHexGrid(
    [
      MAP_PADDING,
      MAP_PADDING,
      MAP_WIDTH - MAP_PADDING,
      MAP_HEIGHT - MAP_PADDING,
    ],
    totalCount,
  );
  const regions = warpedFeatures.map((feature, index) => {
    const id = facetInfo.regions[index].id;
    return {
      centroid: projection(geoCentroid(feature)),
      count: counts[id],
      id,
    };
  });
  const shapes = assignShapes(regions, centers);
  const regionById = new Map(
    facetInfo.regions.map((region) => [region.id, region]),
  );
  const hexagons = shapes.map(({ id, center }, index) => ({
    ...regionById.get(id),
    center,
    id: `${id}-${index}`,
    points: getHexPoints(center, radius),
    regionId: id,
  }));
  const shapeValues = facetInfo.regions.map(
    ({ id, weight }) => weight / counts[id],
  );
  const xValues = shapes.map(({ center }) => center[0]);
  const yValues = shapes.map(({ center }) => center[1]);
  const minX = Math.min(...xValues) - radius;
  const minY = Math.min(...yValues) - radius;
  const maxX = Math.max(...xValues) + radius;
  const maxY = Math.max(...yValues) + radius;
  return {
    boundaryEdges: getHexBoundaryEdges(shapes, radius),
    facetKey: facetInfo.facetKey,
    hexagons,
    labels: getLabels(shapes, regionById, radius),
    radius,
    shapeValueMax: Math.max(...shapeValues),
    shapeValueMin: Math.min(...shapeValues),
    total: facetInfo.regions.reduce((sum, { weight }) => sum + weight, 0),
    viewBox: [minX, minY, maxX - minX, maxY - minY],
  };
}

export default function HexMap({ datumSet }) {
  const { datumList } = datumSet;
  const { regionDimIndex, regionClass, stackDimIndex } =
    getGeoDimInfo(datumList);
  const shapeUnit = `${datumList[0].query.entityClass.name.toLowerCase()}s`;
  const geoJson = useGeoJson(regionClass);

  const { maps, legendItems } = useMemo(() => {
    if (!geoJson) {
      return { maps: [], legendItems: [] };
    }
    console.debug(
      `[HexMap] Preparing map geometry from ${datumList.length} datums`,
    );
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
    const facetGroups = groupDatumListByFacet(datumList, facetDimIndexes);
    console.debug(
      `[HexMap] Matched ${geoFeatures.length}/${geoJson.features.length} geographic features across ${facetGroups.length} facets`,
    );
    const facetInfos = groupDatumListByFacet(datumList, facetDimIndexes).map(
      ({ facetKey, facetDatumList }) =>
        buildFacetInfo(
          geoFeatures,
          facetKey,
          buildFeatureToDataMap(facetDatumList, regionDimIndex, stackDimIndex),
        ),
    );
    const valuePerHexagon = getValuePerShape(
      facetInfos.flatMap(({ regions }) => regions.map(({ weight }) => weight)),
      HEX_MAP_MAX_HEXAGONS,
    );
    const legendItemMap = new Map();
    facetInfos.forEach(({ regions }) =>
      regions.forEach(({ display }) =>
        legendItemMap.set(display.label, {
          id: display.label,
          label: display.label,
          color: display.color,
        }),
      ),
    );
    const maps = DimensionUtils.sortFacets(
      facetInfos.map((facetInfo) =>
        buildHexMapLayout(facetInfo, valuePerHexagon),
      ),
      datumList,
      facetDimIndexes,
      (a, b) => b.total - a.total,
    );
    console.debug(
      `[HexMap] Built ${maps.length} maps with ${maps.reduce((count, map) => count + map.hexagons.length, 0)} hexagons`,
    );
    return { maps, legendItems: Array.from(legendItemMap.values()) };
  }, [geoJson, datumList, regionDimIndex, stackDimIndex]);

  if (!geoJson) {
    return <Typography>Loading hex map…</Typography>;
  }

  return (
    <Box data-testid="hexmaps">
      {maps.length > 1 && <Box data-testid="hexmap-facets" display="none" />}
      <MultiChartLayout
        facets={maps.map((map) => ({ facetKey: map.facetKey, data: map }))}
        xAxisDimName={regionClass.name}
        yAxisLabel=""
        renderChart={({ data }) => (
          <Box
            data-testid="hexmap"
            sx={{
              width: "100%",
              maxWidth: MAP_WIDTH,
              mx: "auto",
              "& svg": { width: "100%", height: "auto", display: "block" },
            }}
          >
            <svg
              viewBox={data.viewBox.join(" ")}
              role="img"
              aria-label="Hex map"
            >
              {data.hexagons.map((hexagon) => (
                <polygon
                  key={hexagon.id}
                  points={hexagon.points.map(([x, y]) => `${x},${y}`).join(" ")}
                  fill={hexagon.display.color ?? MAP_UNKNOWN_COLOR}
                  stroke={MAP_BORDER_COLOR}
                  strokeWidth={HEX_MAP_EDGE_WIDTH}
                >
                  <title>
                    {hexagon.feature.properties.name}: {hexagon.display.label} (
                    {FormatUtils.humanizeValue(hexagon.display.value)})
                  </title>
                </polygon>
              ))}
              <g pointerEvents="none">
                {data.boundaryEdges.map(({ start, end }, index) => (
                  <line
                    key={`${start.join(",")}-${end.join(",")}-${index}`}
                    x1={start[0]}
                    y1={start[1]}
                    x2={end[0]}
                    y2={end[1]}
                    stroke={MAP_BORDER_COLOR}
                    strokeWidth={HEX_MAP_REGION_BORDER_WIDTH}
                  />
                ))}
                {data.labels.map(
                  ({ angle, center, color, fontSize, id, name }) => (
                    <text
                      key={id}
                      x={center[0]}
                      y={center[1]}
                      textAnchor="middle"
                      dominantBaseline="central"
                      fill={
                        FormatUtils.isLightColor(color)
                          ? MAP_LABEL_DARK_COLOR
                          : MAP_LABEL_LIGHT_COLOR
                      }
                      fontSize={fontSize}
                      transform={`rotate(${angle} ${center[0]} ${center[1]})`}
                    >
                      {name}
                    </text>
                  ),
                )}
              </g>
            </svg>
            <Typography
              variant="caption"
              sx={{
                color: HEX_MAP_SCALE_COLOR,
                fontSize: HEX_MAP_SCALE_FONT_SIZE,
              }}
            >
              1 hexagon ={" "}
              {data.shapeValueMax - data.shapeValueMin < 1
                ? FormatUtils.humanizeValue(data.shapeValueMin)
                : `${FormatUtils.humanizeValue(
                    data.shapeValueMin,
                  )} to ${FormatUtils.humanizeValue(data.shapeValueMax)}`}{" "}
              {shapeUnit}
            </Typography>
          </Box>
        )}
      />
      <Legend items={legendItems} />
    </Box>
  );
}

HexMap.IS_CHART = false;
