import { useEffect, useMemo, useState } from "react";
import { feature } from "topojson-client";
import { Box, Typography } from "@mui/material";
import { Choropleth } from "@nivo/geo";
import { geoMercator } from "d3-geo";

import StringUtils from "../../../nonview/base/String.js";
import WWW from "../../../nonview/base/WWW.js";
import Region from "../../../nonview/core/thing/concept/category_concept/region/region/Region.js";
import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
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
        ? FormatUtils.toThingLabel(datum.query.dimThingList[stackDimIndex])
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

export function groupDatumListByFacet(datumList, facetDimIndexes) {
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

function getFeatureRegionId(feature) {
  return feature.properties.id ?? feature.properties.name;
}

export function buildRegionIdToWeight(features, dataMap) {
  const regionIdToWeight = {};
  for (const geoFeature of features) {
    const match = matchFeatureToValue(geoFeature, dataMap);
    if (match) {
      regionIdToWeight[getFeatureRegionId(geoFeature)] = match.items.reduce(
        (total, item) => total + item.value,
        0,
      );
    }
  }
  return regionIdToWeight;
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

function setCartogramViewBox(element) {
  element
    ?.querySelector("svg")
    ?.setAttribute("viewBox", `0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`);
}

export default function Cartogram({ datumSet }) {
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

  const { cartograms, legendItems } = useMemo(() => {
    if (!geoJson) {
      return {
        cartograms: [],
        legendItems: [],
      };
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
    const legendItemMap = new Map();
    const cartograms = groupDatumListByFacet(datumList, facetDimIndexes).map(
      ({ facetKey, facetDatumList }) => {
        const dataMap = buildFeatureToDataMap(
          facetDatumList,
          regionDimIndex,
          stackDimIndex,
        );
        const regionIdToWeight = buildRegionIdToWeight(geoFeatures, dataMap);
        const deformedFeatures = JSON.parse(JSON.stringify(geoFeatures));
        CartogramUtils.compute(deformedFeatures, regionIdToWeight);

        const features = [];
        const data = [];
        for (const geoFeature of deformedFeatures) {
          const match = matchFeatureToValue(geoFeature, dataMap);
          const display = match ? getDisplayItem(match.items) : null;
          const id = String(getFeatureRegionId(geoFeature));
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

        const projection = geoMercator().fitExtent(
          [
            [MAP_PADDING, MAP_PADDING],
            [MAP_WIDTH - MAP_PADDING, MAP_HEIGHT - MAP_PADDING],
          ],
          {
            type: "MultiPoint",
            coordinates: getGeoCoordinates(deformedFeatures),
          },
        );
        const [translateX, translateY] = projection.translate();
        return {
          facetKey,
          features,
          data,
          projectionScale: projection.scale(),
          projectionTranslation: [
            translateX / MAP_WIDTH,
            translateY / MAP_HEIGHT,
          ],
          total: data.reduce((sum, item) => sum + item.value, 0),
        };
      },
    );

    return {
      cartograms: DimensionUtils.sortFacets(
        cartograms,
        datumList,
        facetDimIndexes,
        (a, b) => b.total - a.total,
      ),
      legendItems: Array.from(legendItemMap.values()),
    };
  }, [geoJson, datumList, regionDimIndex, stackDimIndex]);

  if (!geoJson) {
    return <Typography>Loading cartogram…</Typography>;
  }

  const renderCartogram = ({
    features,
    data,
    projectionScale,
    projectionTranslation,
  }) => {
    const maxValue = Math.max(...data.map(({ value }) => value), 1);
    return (
      <Box
        ref={setCartogramViewBox}
        data-testid="cartogram"
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
          role="img"
        />
      </Box>
    );
  };

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
        xAxisDimName={regionClass.name}
        yAxisLabel=""
        renderChart={({ data }) => renderCartogram(data)}
      />
      <Legend items={legendItems} />
    </Box>
  );
}

Cartogram.IS_CHART = false;
