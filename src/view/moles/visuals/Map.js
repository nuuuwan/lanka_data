import { useEffect, useMemo, useState } from "react";
import { feature } from "topojson-client";
import { Box, Typography } from "@mui/material";
import { Choropleth } from "@nivo/geo";
import { geoMercator, geoPath } from "d3-geo";

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
  MAP_UNKNOWN_COLOR,
  MAP_WIDTH,
} from "../../_cons/MapCons.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import Legend from "./Legend.js";

function getRegionDimIndex(datumList) {
  return datumList[0].query.dimThingList.findIndex(
    (thing) => thing instanceof Region,
  );
}

function getStackDimIndex(datumList, regionDimIndex) {
  const { length } = datumList[0].query.dimThingList;
  return Array.from({ length }, (_, i) => i).find(
    (i) =>
      i !== regionDimIndex &&
      new Set(datumList.map((d) => d.query.dimThingList[i].value)).size > 1,
  );
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

  const {
    features,
    data,
    labels,
    legendItems,
    projectionScale,
    projectionTranslation,
  } = useMemo(() => {
    if (!geoJson) {
      return {
        features: [],
        data: [],
        labels: [],
        legendItems: [],
        projectionScale: 0,
        projectionTranslation: [0.5, 0.5],
      };
    }

    const dataMap = buildFeatureToDataMap(
      datumList,
      regionDimIndex,
      stackDimIndex,
    );
    const features = [];
    const data = [];
    for (const geoFeature of geoJson.features) {
      const match = matchFeatureToValue(geoFeature, dataMap);
      const display = match ? getDisplayItem(match.items) : null;
      const id = String(geoFeature.properties.id ?? geoFeature.properties.name);
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
      }
    }

    const legendItems = [];
    const seenLabels = new Set();
    for (const geoFeature of geoJson.features) {
      const match = matchFeatureToValue(geoFeature, dataMap);
      const display = match ? getDisplayItem(match.items) : null;
      if (display && !seenLabels.has(display.label)) {
        seenLabels.add(display.label);
        legendItems.push({
          id: display.label,
          label: display.label,
          color: display.color,
        });
      }
    }

    const projection = geoMercator().fitSize([MAP_WIDTH, MAP_HEIGHT], geoJson);
    const path = geoPath(projection);
    const [translateX, translateY] = projection.translate();
    const labels = features
      .map((geoFeature) => ({
        backgroundColor: geoFeature.fill ?? MAP_UNKNOWN_COLOR,
        id: geoFeature.id,
        name: geoFeature.properties.name,
        position: path.centroid(geoFeature),
      }))
      .filter(({ position }) => position.every(Number.isFinite));

    return {
      features,
      data,
      labels,
      legendItems,
      projectionScale: projection.scale(),
      projectionTranslation: [translateX / MAP_WIDTH, translateY / MAP_HEIGHT],
    };
  }, [geoJson, datumList, regionDimIndex, stackDimIndex]);

  if (!geoJson) {
    return <Typography>Loading map…</Typography>;
  }

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
    <Box>
      <Box
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
          layers={["features", labelsLayer]}
          role="img"
        />
      </Box>
      <Legend items={legendItems} />
    </Box>
  );
}

MapVisual.IS_CHART = false;
