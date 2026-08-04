import StringUtils from "../../../nonview/base/String.js";
import CategoryConcept from "../../../nonview/core/thing/concept/category_concept/CategoryConcept.js";
import Region from "../../../nonview/core/thing/concept/category_concept/region/region/Region.js";
import Thing from "../../../nonview/core/thing/Thing.js";
import DimensionUtils from "./DimensionUtils.js";
import FormatUtils from "./FormatUtils.js";

export function getGeoDimInfo(datumList) {
  const regionDimIndex = datumList[0].query.dimThingList.findIndex(
    (thing) => thing instanceof Region,
  );
  const { varyingDimIndexes } = DimensionUtils.getDimIndexInfo(datumList);
  const nonRegionDimIndexes = varyingDimIndexes.filter(
    (dimIndex) => dimIndex !== regionDimIndex,
  );
  const stackDimIndex = nonRegionDimIndexes.findLast(
    (dimIndex) =>
      datumList[0].query.dimThingList[dimIndex] instanceof CategoryConcept,
  );
  return {
    regionDimIndex,
    regionClass: datumList[0].query.dimThingList[regionDimIndex].constructor,
    stackDimIndex: stackDimIndex ?? null,
  };
}

function getFixedCategoryColor(datumList, regionIndex, stackIndex) {
  if (!datumList.length) {
    return null;
  }
  const { varyingDimIndexes } = DimensionUtils.getDimIndexInfo(datumList);
  const usedIndexes = new Set(
    [regionIndex, stackIndex, ...varyingDimIndexes].filter(
      (index) => index !== null && index !== undefined,
    ),
  );
  const fixedDimThings = datumList[0].query.dimThingList.filter(
    (_, index) => !usedIndexes.has(index),
  );
  const fixedCategoryThings = fixedDimThings.filter(
    (thing) =>
      thing instanceof CategoryConcept &&
      thing.value !== Thing.WILDCARD &&
      thing.value !== Thing.SPECIAL_VALUE_EXCLUDED_SMALL,
  );
  if (fixedCategoryThings.length !== 1) {
    return null;
  }
  return fixedCategoryThings[0].getColor();
}

export function buildFeatureToDataMap(datumList, regionIndex, stackIndex) {
  const fixedCategoryColor = getFixedCategoryColor(
    datumList,
    regionIndex,
    stackIndex,
  );
  const dataMap = new Map();
  for (const datum of datumList) {
    const regionThing = datum.query.dimThingList[regionIndex];
    if (!dataMap.has(regionThing.value)) dataMap.set(regionThing.value, []);
    const stackThing = datum.query.dimThingList[stackIndex];
    dataMap.get(regionThing.value).push({
      label: stackThing ? FormatUtils.toThingLabel(stackThing) : "value",
      value: parseFloat(datum.answerThing.value) || 0,
      color: stackThing
        ? stackThing.getColor()
        : (fixedCategoryColor ?? regionThing.getColor()),
    });
  }
  return dataMap;
}

export function groupDatumListByFacet(datumList, facetDimIndexes) {
  const groups = new Map();
  for (const datum of datumList) {
    const key = DimensionUtils.getFacetKey(datum, facetDimIndexes);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(datum);
  }
  return [...groups].map(([facetKey, facetDatumList]) => ({
    facetKey,
    facetDatumList,
  }));
}

export function matchFeatureToValue(feature, dataMap) {
  const featureName = StringUtils.toSnakeCase(feature.properties.name);
  const compactName = featureName.replace(/_/g, "");
  for (const [regionValue, items] of dataMap) {
    const normalized = StringUtils.toSnakeCase(regionValue);
    if (
      normalized === featureName ||
      normalized.replace(/_/g, "") === compactName
    )
      return { regionValue, items };
  }
  return null;
}

export function getFeatureRegionId(feature) {
  return feature.properties.id ?? feature.properties.name;
}

export function buildGeoVisualData(features, dataMap, legendItemMap) {
  const visualFeatures = [];
  const data = [];
  for (const feature of features) {
    const match = matchFeatureToValue(feature, dataMap);
    const display = match
      ? match.items.reduce((best, item) =>
          item.value > best.value ? item : best,
        )
      : null;
    const id = String(getFeatureRegionId(feature));
    visualFeatures.push({ ...feature, id, fill: display?.color });
    if (display) {
      data.push({ id, value: display.value, categoryLabel: display.label });
      legendItemMap.set(display.label, {
        id: display.label,
        label: display.label,
        color: display.color,
      });
    }
  }
  return { features: visualFeatures, data };
}
