import { useMemo } from "react";
import { Box, Typography } from "@mui/material";
import { GeoMap } from "@nivo/geo";
import { geoCentroid } from "d3-geo";

import useGeoJson from "../../../nonview/base/useGeoJson.js";
import {
  MAP_BORDER_COLOR,
  MAP_BORDER_WIDTH,
  MAP_HEIGHT,
  MAP_PADDING,
  MAP_UNKNOWN_COLOR,
  MAP_WIDTH,
} from "../../_cons/MapCons.js";
import DimensionUtils from "../visual_utils/DimensionUtils.js";
import {
  buildFeatureToDataMap,
  getGeoDimInfo,
  groupDatumListByFacet,
  matchFeatureToValue,
  getProjectionInfo,
} from "../visual_utils/GeoVisualUtils.js";
import MultiChartLayout from "../visual_utils/MultiChartLayout.js";
import Legend from "./Legend.js";

const HEX_AREA_FACTOR = (3 * Math.sqrt(3)) / 2;
const GRID_FACTOR = 1.3;
const MAX_GRID_ITERATIONS = 12;
const MAX_SHAPE_ERROR = 0.1;
const MAX_HEXAGONS = 80;

function getDisplayItem(items) {
  return items.reduce((best, item) => (item.value > best.value ? item : best));
}

export function getHexagonCount(value, maxValue) {
  if (value <= 0 || maxValue <= 0) return 0;
  return Math.max(1, Math.round((value / maxValue) * MAX_HEXAGONS));
}

function getValuePerHexagon(values) {
  const weights = values.filter((value) => value > 0);
  if (!weights.length) return null;
  const maxError = (valuePerHexagon) =>
    Math.max(
      ...weights.map((value) => {
        const ideal = value / valuePerHexagon;
        return Math.abs(Math.max(1, Math.round(ideal)) - ideal) / ideal;
      }),
    );
  const candidates = [Math.min(...weights) * 2 * MAX_SHAPE_ERROR];
  weights.forEach((weight) => {
    for (let count = 1; count <= 7; count += 1) {
      candidates.push((weight * (1 + MAX_SHAPE_ERROR)) / count);
    }
  });
  return (
    candidates
      .filter((candidate) => maxError(candidate) <= MAX_SHAPE_ERROR + 1e-9)
      .sort((a, b) => b - a)[0] ?? candidates[0]
  );
}

function buildGrid(bounds, totalCount) {
  const [minX, minY, maxX, maxY] = bounds;
  const target = Math.max(totalCount * GRID_FACTOR, totalCount + 1);
  const area = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  let radius = Math.sqrt(area / (Math.max(target, 1) * HEX_AREA_FACTOR));
  let centers = [];
  for (let iteration = 0; iteration <= MAX_GRID_ITERATIONS; iteration += 1) {
    const dx = Math.sqrt(3) * radius;
    const dy = 1.5 * radius;
    centers = [];
    for (let row = 0, y = minY; y <= maxY + dy; row += 1, y += dy) {
      for (let x = minX + (row % 2) * (dx / 2); x <= maxX + dx; x += dx) {
        centers.push([x, y]);
      }
    }
    if (centers.length >= totalCount) break;
    radius *= 0.85;
  }
  return { centers, radius };
}

function assignHexagons(features, counts, centers, radius, projection) {
  const slots = [];
  features.forEach((geoFeature, index) => {
    const [x, y] = projection(geoCentroid(geoFeature));
    for (let count = 0; count < counts[index]; count += 1) {
      slots.push({ index, x, y });
    }
  });
  const available = [...centers];
  return slots
    .map((slot) => {
      let bestIndex = 0;
      let bestDistance = Infinity;
      available.forEach(([x, y], index) => {
        const distance = (slot.x - x) ** 2 + (slot.y - y) ** 2;
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = index;
        }
      });
      const center = available.splice(bestIndex, 1)[0];
      return { index: slot.index, center };
    })
    .filter(({ center }) => center)
    .map(({ index, center }) => {
      const [x, y] = center;
      const points = Array.from({ length: 6 }, (_, pointIndex) => {
        const angle = (Math.PI / 3) * pointIndex;
        return projection.invert([
          x + radius * Math.cos(angle),
          y + radius * Math.sin(angle),
        ]);
      });
      return {
        index,
        type: "Feature",
        properties: { type: "hex" },
        geometry: { type: "Polygon", coordinates: [[...points, points[0]]] },
      };
    });
}

export default function HexMap({ datumSet }) {
  const { datumList } = datumSet;
  const regionDimIndex = getRegionDimIndex(datumList);
  const stackDimIndex = getStackDimIndex(datumList, regionDimIndex);
  const regionClass =
    datumList[0].query.dimThingList[regionDimIndex].constructor;
  const [geoJson, setGeoJson] = useState(null);

  useEffect(() => {
    async function load() {
      const geoUrl = regionClass.getGeoURL();
      console.debug(
        `[HexMap] Loading geography for ${regionClass.name} from ${geoUrl}`,
      );
      const topoJson = await WWW.json(geoUrl);
      const nextGeoJson = feature(topoJson, topoJson.objects.data);
      console.debug(
        `[HexMap] Loaded ${nextGeoJson.features.length} geographic features for ${regionClass.name}`,
      );
      setGeoJson(nextGeoJson);
    }
    load();
  }, [regionClass]);

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
    const projection = geoMercator().fitExtent(
      [
        [MAP_PADDING, MAP_PADDING],
        [MAP_WIDTH - MAP_PADDING, MAP_HEIGHT - MAP_PADDING],
      ],
      geoJson,
    );
    const legendItemMap = new Map();
    const maps = facetGroups
      .map(({ facetKey, facetDatumList }) => {
        const dataMap = buildFeatureToDataMap(
          facetDatumList,
          regionDimIndex,
          stackDimIndex,
        );
        const displays = geoFeatures.map((geoFeature) => {
          const match = matchFeatureToValue(geoFeature, dataMap);
          return match ? getDisplayItem(match.items) : null;
        });
        const counts = displays.map((item) =>
          item && valuePerHexagon
            ? Math.max(1, Math.round(item.value / valuePerHexagon))
            : 0,
        );
        const { centers, radius } = buildGrid(
          [
            MAP_PADDING,
            MAP_PADDING,
            MAP_WIDTH - MAP_PADDING,
            MAP_HEIGHT - MAP_PADDING,
          ],
          counts.reduce((sum, count) => sum + count, 0),
        );
        const hexFeatures = assignHexagons(
          geoFeatures,
          counts,
          centers,
          radius,
          projection,
        );
        hexFeatures.forEach((hexagon) => {
          const display = displays[hexagon.index];
          const geoFeature = geoFeatures[hexagon.index];
          if (display) {
            hexagon.properties = {
              ...hexagon.properties,
              color: display.color,
              name: geoFeature.properties.name,
              value: display.value,
              label: display.label,
            };
          }
          if (display) {
            legendItemMap.set(display.label, {
              id: display.label,
              label: display.label,
              color: display.color,
            });
          }
        });
        return {
          facetKey,
          features: [...geoFeatures, ...hexFeatures],
          projectionScale,
          projectionTranslation,
          total: displays.reduce((sum, item) => sum + (item?.value ?? 0), 0),
        };
      })
      .sort((a, b) => b.total - a.total);
    console.debug(
      `[HexMap] Built ${maps.length} maps with ${maps.reduce((count, map) => count + map.features.length, 0)} total features`,
    );
    return { maps, legendItems: Array.from(legendItemMap.values()) };
  }, [geoJson, datumList, regionDimIndex, stackDimIndex]);

  useEffect(() => {
    if (geoJson) {
      console.debug(
        `[HexMap] Prepared ${maps.length} maps with ${legendItems.length} legend items from ${datumList.length} datums`,
      );
    }
  }, [geoJson, maps, legendItems, datumList.length]);

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
            <GeoMap
              width={MAP_WIDTH}
              height={MAP_HEIGHT}
              features={data.features}
              projectionType="mercator"
              projectionScale={data.projectionScale}
              projectionTranslation={data.projectionTranslation}
              fillColor={(mapFeature) =>
                mapFeature.properties.type === "hex"
                  ? mapFeature.properties.color
                  : MAP_UNKNOWN_COLOR
              }
              borderWidth={MAP_BORDER_WIDTH}
              borderColor={MAP_BORDER_COLOR}
              role="img"
            />
          </Box>
        )}
      />
      <Legend items={legendItems} />
    </Box>
  );
}

HexMap.IS_CHART = false;
