import Time from "./thing/concept/atoms/Time.js";
import Region from "./thing/concept/category_concept/region/region/Region.js";
import {
  getDimensionLabel,
  getValueLabels,
  isSpecific,
  joinLabels,
} from "./VisualMetadataLabels.js";

export function getGeographyLabel(query, dimensions) {
  const regionDimensions = dimensions.filter(
    (thing) => thing instanceof Region,
  );
  const constrainedRegions = (query.parentRegionConstraintList ?? [])
    .map(({ parentRegion }) => parentRegion)
    .filter(Boolean);
  const specificRegions = [...regionDimensions, ...constrainedRegions].filter(
    isSpecific,
  );
  const varyingRegions = regionDimensions.filter((thing) => !isSpecific(thing));

  const locations = specificRegions.map(
    (thing) =>
      `${joinLabels(getValueLabels(thing))} ${getDimensionLabel(thing)}`,
  );
  const levels = varyingRegions.map((thing) => getDimensionLabel(thing));

  if (levels.length && locations.length) {
    return `${joinLabels(levels)} in ${joinLabels(locations)}`;
  }
  if (locations.length) {
    return joinLabels(locations);
  }
  if (levels.length) {
    return `all available ${joinLabels(levels)}`;
  }
  return "all available geographies";
}

export function getTimePeriodLabel(dimensions) {
  const timeDimensions = dimensions.filter((thing) => thing instanceof Time);
  if (
    !timeDimensions.length ||
    timeDimensions.some((thing) => !isSpecific(thing))
  ) {
    return "all available periods";
  }
  return joinLabels(timeDimensions.flatMap(getValueLabels));
}

export function getFilterLabel(query, dimensions) {
  const filters = dimensions
    .filter(isSpecific)
    .map(
      (thing) =>
        `${getDimensionLabel(thing)}: ${joinLabels(getValueLabels(thing))}`,
    );
  for (const { parentRegion } of query.parentRegionConstraintList ?? []) {
    if (parentRegion && isSpecific(parentRegion)) {
      filters.push(
        `${getDimensionLabel(parentRegion)}: ${joinLabels(
          getValueLabels(parentRegion),
        )}`,
      );
    }
  }
  return filters.length ? filters.join("; ") : "none";
}
