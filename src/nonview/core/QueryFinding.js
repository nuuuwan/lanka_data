import Thing from "./thing/Thing.js";

const ENTITY_LABELS = {
  Census: "census records",
  House: "households",
  Person: "population",
  Vote: "votes",
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

function humanizeIdentifier(value) {
  return value
    .replace(/([a-z\d])([A-Z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/(\D)(\d)/g, "$1 $2")
    .replaceAll("_", " ")
    .toLowerCase();
}

function getDimensionLabel(thing) {
  const className = thing.constructor.getClassName();
  return DIMENSION_LABELS[className] ?? humanizeIdentifier(className);
}

function getThingValues(thing) {
  return (thing.valueList ?? [thing.value]).map((value) =>
    thing.constructor.fromValue(value).getLabel(),
  );
}

function joinLabels(labels) {
  if (labels.length < 2) {
    return labels[0] ?? "";
  }
  return `${labels.slice(0, -1).join(", ")} and ${labels.at(-1)}`;
}

function getConstraintText(thing) {
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

function getAggregateLabel(query) {
  const entityClassName = query.entityClass.getClassName();
  if (query.aggregate.toLowerCase() === "count") {
    return (
      { Person: "population", Vote: "valid votes" }[entityClassName] ?? "count"
    );
  }
  return humanizeIdentifier(query.aggregate);
}

export default function getQueryFinding(query) {
  const entityClassName = query.entityClass.getClassName();
  const entityLabel =
    ENTITY_LABELS[entityClassName] ?? `${humanizeIdentifier(entityClassName)}s`;
  const aggregateLabel = getAggregateLabel(query);
  const groupLabels = query.dimThingList
    .filter((thing) => thing.value === Thing.WILDCARD)
    .map(getDimensionLabel);
  const constraints = query.dimThingList
    .filter((thing) => thing.value !== Thing.WILDCARD)
    .map(getConstraintText);
  constraints.push(
    ...(query.parentRegionConstraintList ?? [])
      .map(({ parentRegion }) => parentRegion)
      .filter(Boolean)
      .map(getConstraintText),
  );

  const groupedBy =
    groupLabels.length > 0 ? ` by ${joinLabels(groupLabels)}` : "";
  const constrainedBy =
    constraints.length > 0 ? ` ${constraints.join(" and ")}` : "";
  const finding = `${aggregateLabel} of ${entityLabel}${groupedBy}${constrainedBy}`;
  return finding.charAt(0).toUpperCase() + finding.slice(1);
}
