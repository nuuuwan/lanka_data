import { useEffect, useMemo, useState } from "react";
import { feature } from "topojson-client";
import { Box, Typography } from "@mui/material";
import { Choropleth } from "@nivo/geo";
import { geoMercator } from "d3-geo";

import StringUtils from "../../../nonview/base/String.js";
import WWW from "../../../nonview/base/WWW.js";
import Region from "../../../nonview/core/thing/concept/category_concept/region/region/Region.js";
import {
  MAP_BORDER_COLOR,
  MAP_BORDER_WIDTH,
  MAP_HEIGHT,
  MAP_LABEL_DARK_COLOR,
  MAP_LABEL_FONT_SIZE,
  MAP_LABEL_LIGHT_COLOR,
  MAP_MAX_LABEL_COUNT,
  MAP_PADDING,
  MAP_UNKNOWN_COLOR,
  MAP_WIDTH,
} from "../../_cons/MapCons.js";
import DimensionUtils from "../visual_utils/DimensionUtils.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import MultiChartLayout from "../visual_utils/MultiChartLayout.js";
import Legend from "./Legend.js";

function getRegionDimIndex(datumList) {
  return datumList[0].query.dimThingList.findIndex(
    (thing) => thing instanceof Region,
  );
}

function getStackDimIndex(datumList, regionDimIndex) {
  const { varyingDimIndexes } = DimensionUtils.getDimIndexInfo(datumList);
  return varyingDimIndexes.filter((i) => i !== regionDimIndex).at(-1);
}

function getRegionClass(datumList, regionDimIndex) {
  return datumList[0].query.dimThingList[regionDimIndex].constructor;
}

function buildFeatureToDataMap(datumList, regionDimIndex, stackDimIndex) {
  const map = new Map();
  for (const datum of datumList) {
    const regionValue = datum.query.dimThingList[regionDimIndex].value;
    if (!map.has(regionValue)) {
      map.set(regionValue, []);
    }
    const stackLabel =
      stackDimIndex !== undefined
        ? FormatUtils.toTitleCase(datum.query.dimThingList[stackDimIndex].value)
        : "value";
    const color =
      stackDimIndex !== undefined
        ? datum.query.dimThingList[stackDimIndex].getColor()
        : datum.query.dimThingList[regionDimIndex].getColor();
    map.get(regionValue).push({
      label: stackLabel,
      value: parseFloat(datum.answerThing.value) || 0,
      color,
    });
  }
  return map;
}

function getDisplayItem(items) {
  if (items.length === 1) {
    return items[0];
  }
  return items.reduce((best, item) => (item.value > best.value ? item : best));
}

function groupDatumListByFacet(datumList, facetDimIndexes) {
  const groups = new Map();
  for (const datum of datumList) {
    const facetKey = DimensionUtils.getFacetKey(datum, facetDimIndexes);
    if (!groups.has(facetKey)) {
      groups.set(facetKey, []);
    }
    groups.get(facetKey).push(datum);
  }
  return Array.from(groups.entries()).map(([facetKey, facetDatumList]) => ({
    facetKey,
    facetDatumList,
  }));
}

function getGeoCoordinates(features) {
  const coordinates = [];

  function collect(value) {
    if (!Array.isArray(value)) {
      return;
    }
    if (typeof value[0] === "number") {
      coordinates.push(value);
      return;
    }
    value.forEach(collect);
  }

  features.forEach(({ geometry }) => collect(geometry.coordinates));
  return coordinates;
}

function getFeatureCenter(feature, projection) {
  const bounds = getGeoCoordinates([feature])
    .map(projection)
    .reduce(
      ([minX, minY, maxX, maxY], [x, y]) => [
        Math.min(minX, x),
        Math.min(minY, y),
        Math.max(maxX, x),
        Math.max(maxY, y),
      ],
      [Infinity, Infinity, -Infinity, -Infinity],
    );
  return [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2];
}

function setMapViewBox(element) {
  element
    ?.querySelector("svg")
    ?.setAttribute("viewBox", `0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`);
}

export function matchFeatureToValue(feature, dataMap) {
  const featureName = StringUtils.toSnakeCase(feature.properties.name);
  const compactFeatureName = featureName.replace(/_/g, "");
  for (const [regionValue, items] of dataMap) {
    const normalizedRegionValue = StringUtils.toSnakeCase(regionValue);
    if (
      normalizedRegionValue === featureName ||
      normalizedRegionValue.replace(/_/g, "") === compactFeatureName
    ) {
      return { regionValue, items };
    }
  }
  return null;
}

export default function MapVisual({ datumSet }) {
  const { datumList } = datumSet;
  const regionDimIndex = getRegionDimIndex(datumList);
  const regionClass = getRegionClass(datumList, regionDimIndex);
  const stackDimIndex = getStackDimIndex(datumList, regionDimIndex);

  const [geoJson, setGeoJson] = useState(null);
  useEffect(() => {
    async function load() {
      const topoJson = await WWW.json(regionClass.getGeoURL());
      setGeoJson(feature(topoJson, topoJson.objects.data));
    }
    load();
  }, [regionClass]);

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
      const projection = geoMercator().fitExtent(
        [
          [MAP_PADDING, MAP_PADDING],
          [MAP_WIDTH - MAP_PADDING, MAP_HEIGHT - MAP_PADDING],
        ],
        {
          type: "MultiPoint",
          coordinates: getGeoCoordinates(geoJson.features),
        },
      );
      const [translateX, translateY] = projection.translate();
      const legendItemMap = new Map();
      const maps = groupDatumListByFacet(datumList, facetDimIndexes)
        .map(({ facetKey, facetDatumList }) => {
          const dataMap = buildFeatureToDataMap(
            facetDatumList,
            regionDimIndex,
            stackDimIndex,
          );
          const features = [];
          const data = [];
          for (const geoFeature of geoJson.features) {
            const match = matchFeatureToValue(geoFeature, dataMap);
            const display = match ? getDisplayItem(match.items) : null;
            const id = String(
              geoFeature.properties.id ?? geoFeature.properties.name,
            );
            features.push({
              ...geoFeature,
              id,
              fill: display?.color,
            });
            if (display) {
              data.push({
                id,
                value: display.value,
                categoryLabel: display.label,
              });
              legendItemMap.set(display.label, {
                id: display.label,
                label: display.label,
                color: display.color,
              });
            }
          }
          const labels =
            features.length <= MAP_MAX_LABEL_COUNT
              ? features
                  .map((geoFeature) => ({
                    backgroundColor: geoFeature.fill ?? MAP_UNKNOWN_COLOR,
                    id: geoFeature.id,
                    name: geoFeature.properties.name,
                    position: getFeatureCenter(geoFeature, projection),
                  }))
                  .filter(({ position }) => position.every(Number.isFinite))
              : [];
          return {
            facetKey,
            features,
            data,
            labels,
            total: data.reduce((sum, item) => sum + item.value, 0),
          };
        })
        .sort((a, b) => b.total - a.total);

      return {
        maps,
        legendItems: Array.from(legendItemMap.values()),
        projectionScale: projection.scale(),
        projectionTranslation: [
          translateX / MAP_WIDTH,
          translateY / MAP_HEIGHT,
        ],
      };
    }, [geoJson, datumList, regionDimIndex, stackDimIndex]);

  if (!geoJson) {
    return <Typography>Loading map…</Typography>;
  }

  const renderMap = ({ features, data, labels }) => {
    const maxValue = Math.max(...data.map(({ value }) => value), 1);
    const labelsLayer = () => (
      <g data-testid="map-labels" pointerEvents="none">
        {labels.map(({ backgroundColor, id, name, position: [x, y] }) => {
          const lightBackground = FormatUtils.isLightColor(backgroundColor);
          return (
            <text
              key={id}
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="central"
              fill={
                lightBackground ? MAP_LABEL_DARK_COLOR : MAP_LABEL_LIGHT_COLOR
              }
              fontSize={MAP_LABEL_FONT_SIZE}
            >
              {name}
            </text>
          );
        })}
      </g>
    );

    return (
      <Box
        ref={setMapViewBox}
        data-testid="map"
        sx={{
          width: "100%",
          maxWidth: MAP_WIDTH,
          mx: "auto",
          "& svg": {
            width: "100%",
            height: "auto",
            display: "block",
          },
        }}
      >
        <Choropleth
          width={MAP_WIDTH}
          height={MAP_HEIGHT}
          features={features}
          data={data}
          domain={[0, maxValue]}
          label={(mapFeature) =>
            mapFeature.data
              ? `${mapFeature.properties.name}: ${mapFeature.data.categoryLabel}`
              : mapFeature.properties.name
          }
          valueFormat={FormatUtils.humanizeValue}
          projectionType="mercator"
          projectionScale={projectionScale}
          projectionTranslation={projectionTranslation}
          colors={[MAP_UNKNOWN_COLOR, MAP_UNKNOWN_COLOR]}
          unknownColor={MAP_UNKNOWN_COLOR}
          borderWidth={MAP_BORDER_WIDTH}
          borderColor={MAP_BORDER_COLOR}
          layers={labels.length > 0 ? ["features", labelsLayer] : ["features"]}
          role="img"
        />
      </Box>
    );
  };

  return (
    <Box data-testid="maps">
      {maps.length > 1 && <Box data-testid="map-facets" display="none" />}
      <MultiChartLayout
        facets={maps.map((map) => ({
          facetKey: map.facetKey,
          data: map,
        }))}
        xAxisDimName={regionClass.name}
        yAxisLabel=""
        renderChart={({ data }) => renderMap(data)}
      />
      <Legend items={legendItems} />
    </Box>
  );
}

MapVisual.IS_CHART = false;
