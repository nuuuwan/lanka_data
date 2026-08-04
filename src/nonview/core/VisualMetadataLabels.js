import Thing from "./thing/Thing.js";

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

export function humanize(value) {
  return String(value)
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replaceAll("_", " ")
    .toLowerCase();
}

export function joinLabels(labels) {
  if (labels.length < 2) {
    return labels[0] ?? "";
  }
  if (labels.length === 2) {
    return labels.join(" and ");
  }
  return `${labels.slice(0, -1).join(", ")}, and ${labels.at(-1)}`;
}

export function getDimensionLabel(thing) {
  const className = thing.constructor.getClassName();
  return DIMENSION_LABELS[className] ?? humanize(className);
}

export function getValueLabels(thing) {
  return (thing.valueList ?? [thing.value]).map((value) => {
    if (value === thing.value) {
      return humanize(thing.getLabel());
    }
    return humanize(thing.constructor.fromValue(value).getLabel());
  });
}

export function isSpecific(thing) {
  return thing.value !== Thing.WILDCARD;
}

export function getRequestedDimensions(query) {
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

export function getPopulationLabel(query) {
  const entityName = query.entityClass.getClassName();
  return ENTITY_LABELS[entityName] ?? `${humanize(entityName)} records`;
}

export function getTitlePopulationLabel(query) {
  const entityName = query.entityClass.getClassName();
  return TITLE_ENTITY_LABELS[entityName] ?? getPopulationLabel(query);
}
