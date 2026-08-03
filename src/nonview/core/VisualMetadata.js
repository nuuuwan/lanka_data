import Thing from "./thing/Thing.js";
import Time from "./thing/concept/atoms/Time.js";
import Region from "./thing/concept/category_concept/region/region/Region.js";

const ENTITY_LABELS = {
  Census: "censuses",
  House: "houses",
  Person: "people",
  Vote: "votes",
};

const TITLE_ENTITY_LABELS = {
  House: "households",
  Person: "population",
};

const DIMENSION_LABELS = {
  DSD: "divisional secretariat division",
  ED: "electoral district",
  GND: "grama niladhari division",
  PD: "polling division",
};

function humanize(value) {
  return String(value)
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replaceAll("_", " ")
    .toLowerCase();
}

function joinLabels(labels) {
  if (labels.length < 2) {
    return labels[0] ?? "";
  }
  if (labels.length === 2) {
    return labels.join(" and ");
  }
  return `${labels.slice(0, -1).join(", ")}, and ${labels.at(-1)}`;
}

function getDimensionLabel(thing) {
  const className = thing.constructor.getClassName();
  return DIMENSION_LABELS[className] ?? humanize(className);
}

function getValueLabels(thing) {
  return (thing.valueList ?? [thing.value]).map((value) => {
    if (value === thing.value) {
      return humanize(thing.getLabel());
    }
    return humanize(thing.constructor.fromValue(value).getLabel());
  });
}

function isSpecific(thing) {
  return thing.value !== Thing.WILDCARD;
}

function getRequestedDimensions(query) {
  const dimensions = [...query.dimThingList];
  for (const subRegionThing of query.subRegionDimThingList ?? []) {
    if (!isSpecific(subRegionThing)) {
      continue;
    }
    const parentClassName =
      subRegionThing.constructor.getParentRegionInfo().parentClassName;
    const parentIndex = dimensions.findIndex(
      (thing) => thing.constructor.getClassName() === parentClassName,
    );
    if (parentIndex === -1) {
      dimensions.push(subRegionThing);
    } else {
      dimensions[parentIndex] = subRegionThing;
    }
  }
  return dimensions;
}

function getPopulationLabel(query) {
  const entityName = query.entityClass.getClassName();
  return ENTITY_LABELS[entityName] ?? `${humanize(entityName)} records`;
}

function getTitlePopulationLabel(query) {
  const entityName = query.entityClass.getClassName();
  return TITLE_ENTITY_LABELS[entityName] ?? getPopulationLabel(query);
}

function getGeographyLabel(query, dimensions) {
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

function getTimePeriodLabel(dimensions) {
  const timeDimensions = dimensions.filter((thing) => thing instanceof Time);
  if (
    !timeDimensions.length ||
    timeDimensions.some((thing) => !isSpecific(thing))
  ) {
    return "all available periods";
  }
  return joinLabels(timeDimensions.flatMap(getValueLabels));
}

function getFilterLabel(query, dimensions) {
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

function getUnitsLabel(query, datumSet, populationLabel) {
  if (query.aggregate.toLowerCase() === "count") {
    return populationLabel;
  }
  const answerClassName =
    datumSet.datumList[0]?.answerThing?.constructor.getClassName?.();
  if (answerClassName === "Percent") {
    return "percent";
  }
  return humanize(query.aggregate);
}

export default class VisualMetadata {
  static from(query, datumSet) {
    const dimensions = getRequestedDimensions(query);
    const population = getPopulationLabel(query);
    const titlePopulation = getTitlePopulationLabel(query);
    const measure = humanize(query.aggregate);
    return {
      title: `${measure.charAt(0).toUpperCase()}${measure.slice(1)} of ${titlePopulation}`,
      subtitle: [
        `Population: ${population}`,
        `Geography: ${getGeographyLabel(query, dimensions)}`,
        `Time period: ${getTimePeriodLabel(dimensions)}`,
        `Units: ${getUnitsLabel(query, datumSet, population)}`,
        `Filters: ${getFilterLabel(query, dimensions)}`,
      ].join(" • "),
    };
  }
}
