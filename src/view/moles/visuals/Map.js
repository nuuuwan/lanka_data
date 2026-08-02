import { useEffect, useMemo, useState } from "react";
import { feature } from "topojson-client";
import { Box, Typography } from "@mui/material";
import { geoPath, geoMercator } from "d3-geo";

import { FONT_FAMILY } from "../../../AppTheme.js";
import WWW from "../../../nonview/base/WWW.js";
import Region from "../../../nonview/core/thing/concept/category_concept/region/region/Region.js";
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
    (i) => i !== regionDimIndex && new Set(datumList.map((d) => d.query.dimThingList[i].value)).size > 1,
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

function matchFeatureToValue(feature, dataMap) {
  const featureName = feature.properties.name;
  for (const [regionValue, items] of dataMap) {
    if (
      regionValue === featureName.toLowerCase().replace(/\s+/g, "_") ||
      regionValue === featureName.toLowerCase().replace(/\s+/g, "") ||
      regionValue === featureName.toLowerCase()
    ) {
      return { regionValue, items };
    }
  }
  return null;
}

export default function Map({ datumSet }) {
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

  const { dataWithDisplay, legendItems, projection, path } = useMemo(() => {
    if (!geoJson) {
      return {
        dataWithDisplay: [],
        legendItems: [],
        projection: null,
        path: null,
      };
    }

    const dataMap = buildFeatureToDataMap(datumList, regionDimIndex, stackDimIndex);
    const matched = [];
    for (const geoFeature of geoJson.features) {
      const match = matchFeatureToValue(geoFeature, dataMap);
      const display = match ? getDisplayItem(match.items) : null;
      matched.push({ geoFeature, display });
    }

    const legendItems = [];
    const seenLabels = new Set();
    for (const { display } of matched) {
      if (display && !seenLabels.has(display.label)) {
        seenLabels.add(display.label);
        legendItems.push({
          id: display.label,
          label: display.label,
          color: display.color,
        });
      }
    }

    const projection = geoMercator().fitSize([600, 800], geoJson);
    const path = geoPath().projection(projection);

    return { dataWithDisplay: matched, legendItems, projection, path };
  }, [geoJson, datumList, regionDimIndex, stackDimIndex]);

  if (!geoJson || !path) {
    return <Typography>Loading map…</Typography>;
  }

  return (
    <Box>
      <Box
        component="svg"
        viewBox="0 0 600 800"
        sx={{
          width: "100%",
          maxWidth: 600,
          height: "auto",
          display: "block",
          mx: "auto",
        }}
      >
        {dataWithDisplay.map(({ geoFeature, display }) => (
          <path
            key={geoFeature.properties.id}
            d={path(geoFeature)}
            fill={display?.color ?? "#e0e0e0"}
            stroke="#ffffff"
            strokeWidth={0.5}
          >
            <title>
              {geoFeature.properties.name}
              {display
                ? `: ${display.label} ${FormatUtils.humanizeValue(display.value)}`
                : ""}
            </title>
          </path>
        ))}
      </Box>
      <Legend items={legendItems} />
    </Box>
  );
}

Map.IS_CHART = false;
