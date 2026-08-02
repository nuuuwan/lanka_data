import { useEffect, useMemo, useState } from "react";
import { feature } from "topojson-client";
import { Box, Typography } from "@mui/material";
import { GeoMap } from "@nivo/geo";
import { geoContains, geoMercator } from "d3-geo";

import StringUtils from "../../../nonview/base/String.js";
import WWW from "../../../nonview/base/WWW.js";
import Region from "../../../nonview/core/thing/concept/category_concept/region/region/Region.js";
import {
  MAP_BORDER_COLOR,
  MAP_BORDER_WIDTH,
  MAP_HEIGHT,
  MAP_PADDING,
  MAP_UNKNOWN_COLOR,
  MAP_WIDTH,
} from "../../_cons/MapCons.js";
import DimensionUtils from "../visual_utils/DimensionUtils.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import MultiChartLayout from "../visual_utils/MultiChartLayout.js";
import Legend from "./Legend.js";

const HEX_RADIUS = 5;
const HEX_SPACING_X = HEX_RADIUS * 1.75;
const HEX_SPACING_Y = HEX_RADIUS * 1.52;
const MAX_HEXAGONS = 80;

function getRegionDimIndex(datumList) {
  return datumList[0].query.dimThingList.findIndex(
    (thing) => thing instanceof Region,
  );
}

function getStackDimIndex(datumList, regionDimIndex) {
  const { varyingDimIndexes } = DimensionUtils.getDimIndexInfo(datumList);
  return varyingDimIndexes.filter((i) => i !== regionDimIndex).at(-1);
}

function buildFeatureToDataMap(datumList, regionDimIndex, stackDimIndex) {
  const map = new Map();
  for (const datum of datumList) {
    const regionValue = datum.query.dimThingList[regionDimIndex].value;
    const items = map.get(regionValue) ?? [];
    const stackThing =
      stackDimIndex === undefined
        ? datum.query.dimThingList[regionDimIndex]
        : datum.query.dimThingList[stackDimIndex];
    items.push({
      label:
        stackDimIndex === undefined
          ? "value"
          : FormatUtils.toThingLabel(stackThing),
      value: parseFloat(datum.answerThing.value) || 0,
      color: stackThing.getColor(),
    });
    map.set(regionValue, items);
  }
  return map;
}

function getDisplayItem(items) {
  return items.reduce((best, item) => (item.value > best.value ? item : best));
}

function matchFeatureToValue(geoFeature, dataMap) {
  const featureName = StringUtils.toSnakeCase(geoFeature.properties.name);
  const compactFeatureName = featureName.replace(/_/g, "");
  for (const [regionValue, items] of dataMap) {
    const normalized = StringUtils.toSnakeCase(regionValue);
    if (
      normalized === featureName ||
      normalized.replace(/_/g, "") === compactFeatureName
    ) {
      return { regionValue, items };
    }
  }
  return null;
}

function getFeatureCenter(geoFeature, projection) {
  const [minX, minY, maxX, maxY] = geoFeature.geometry.coordinates
    .flat(Infinity)
    .reduce(
      (bounds, value, index, values) =>
        index % 2 === 0
          ? [
              Math.min(bounds[0], projection([value, values[index + 1]])[0]),
              Math.min(bounds[1], projection([value, values[index + 1]])[1]),
              Math.max(bounds[2], projection([value, values[index + 1]])[0]),
              Math.max(bounds[3], projection([value, values[index + 1]])[1]),
            ]
          : bounds,
      [Infinity, Infinity, -Infinity, -Infinity],
    );
  return [(minX + maxX) / 2, (minY + maxY) / 2];
}

function createHexagon(center, projection) {
  const [x, y] = center;
  const points = Array.from({ length: 6 }, (_, index) => {
    const angle = (Math.PI / 3) * index;
    return projection.invert([
      x + HEX_RADIUS * Math.cos(angle),
      y + HEX_RADIUS * Math.sin(angle),
    ]);
  });
  return {
    type: "Feature",
    properties: { type: "hex" },
    geometry: { type: "Polygon", coordinates: [[...points, points[0]]] },
  };
}

export function getHexagonCount(value, maxValue) {
  if (value <= 0 || maxValue <= 0) {
    return 0;
  }
  return Math.max(1, Math.round((value / maxValue) * MAX_HEXAGONS));
}

function createHexagons(geoFeature, count, projection) {
  if (count === 0) {
    return [];
  }
  const [centerX, centerY] = getFeatureCenter(geoFeature, projection);
  const candidates = [];
  for (
    let y = centerY - MAP_HEIGHT / 2;
    y <= centerY + MAP_HEIGHT / 2;
    y += HEX_SPACING_Y
  ) {
    for (
      let x = centerX - MAP_WIDTH / 2;
      x <= centerX + MAP_WIDTH / 2;
      x += HEX_SPACING_X
    ) {
      const candidate = [
        x + (Math.round(y / HEX_SPACING_Y) % 2) * (HEX_SPACING_X / 2),
        y,
      ];
      const geographicPoint = projection.invert(candidate);
      if (geographicPoint && geoContains(geoFeature, geographicPoint)) {
        candidates.push(candidate);
      }
    }
  }
  candidates.sort(
    (a, b) =>
      (a[0] - centerX) ** 2 +
      (a[1] - centerY) ** 2 -
      ((b[0] - centerX) ** 2 + (b[1] - centerY) ** 2),
  );
  return candidates
    .slice(0, count)
    .map((candidate) => createHexagon(candidate, projection));
}

function groupDatumListByFacet(datumList, facetDimIndexes) {
  const groups = new Map();
  for (const datum of datumList) {
    const facetKey = DimensionUtils.getFacetKey(datum, facetDimIndexes);
    groups.set(facetKey, [...(groups.get(facetKey) ?? []), datum]);
  }
  return Array.from(groups, ([facetKey, facetDatumList]) => ({
    facetKey,
    facetDatumList,
  }));
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
    const projection = geoMercator().fitExtent(
      [
        [MAP_PADDING, MAP_PADDING],
        [MAP_WIDTH - MAP_PADDING, MAP_HEIGHT - MAP_PADDING],
      ],
      geoJson,
    );
    const legendItemMap = new Map();
    const maps = groupDatumListByFacet(datumList, facetDimIndexes)
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
        const maxValue = Math.max(
          ...displays.map((item) => item?.value ?? 0),
          1,
        );
        const hexFeatures = [];
        geoFeatures.forEach((geoFeature, index) => {
          const display = displays[index];
          if (!display) return;
          const hexagons = createHexagons(
            geoFeature,
            getHexagonCount(display.value, maxValue),
            projection,
          );
          hexagons.forEach((hexagon) => {
            hexagon.properties = {
              ...hexagon.properties,
              color: display.color,
              name: geoFeature.properties.name,
              value: display.value,
              label: display.label,
            };
          });
          hexFeatures.push(...hexagons);
          legendItemMap.set(display.label, {
            id: display.label,
            label: display.label,
            color: display.color,
          });
        });
        return {
          facetKey,
          features: [...geoFeatures, ...hexFeatures],
          projectionScale: projection.scale(),
          projectionTranslation: [
            projection.translate()[0] / MAP_WIDTH,
            projection.translate()[1] / MAP_HEIGHT,
          ],
          total: displays.reduce((sum, item) => sum + (item?.value ?? 0), 0),
        };
      })
      .sort((a, b) => b.total - a.total);
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
