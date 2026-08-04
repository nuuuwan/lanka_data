export const ENTITY_LABELS = {
  Census: "census records",
  House: "households",
  Person: "population",
  Vote: "votes",
};

export const COUNT_LABELS = {
  Person: "population",
  Vote: "valid votes",
};

const DIMENSION_LABELS = {
  DSD: "divisional secretariat division",
  ED: "electoral district",
  GND: "Grama Niladhari division",
  PD: "polling division",
};

const REGION_DIMENSIONS = new Set([
  "Country",
  "District",
  "DSD",
  "ED",
  "GND",
  "PD",
  "Province",
]);

export function humanizeIdentifier(value) {
  return value
    .replace(/([a-z\d])([A-Z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/(\D)(\d)/g, "$1 $2")
    .replaceAll("_", " ")
    .toLowerCase();
}

export function getDimensionLabel(thing) {
  const className = thing.constructor.getClassName();
  return DIMENSION_LABELS[className] ?? humanizeIdentifier(className);
}

export function getThingValues(thing) {
  return (thing.valueList ?? [thing.value]).map((value) =>
    thing.constructor.fromValue(value).getLabel(),
  );
}

export function joinLabels(labels) {
  if (labels.length < 2) {
    return labels[0] ?? "";
  }
  return `${labels.slice(0, -1).join(", ")} and ${labels.at(-1)}`;
}

export function getConstraintText(thing) {
  const className = thing.constructor.getClassName();
  const values = joinLabels(getThingValues(thing));
  if (className === "Time") {
    return `in ${values}`;
  }
  if (REGION_DIMENSIONS.has(className)) {
    return `in ${values} ${getDimensionLabel(thing)}`;
  }
  return `where ${getDimensionLabel(thing)} is ${values}`;
}

export function getAggregateLabel(query) {
  const entityClassName = query.entityClass.getClassName();
  if (query.aggregate.toLowerCase() === "count") {
    return COUNT_LABELS[entityClassName] ?? "count";
  }
  return humanizeIdentifier(query.aggregate);
}
