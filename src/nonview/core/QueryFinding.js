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

function titleCase(value) {
  return humanizeIdentifier(value).replace(/\b\w/g, (character) =>
    character.toUpperCase(),
  );
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

function getValueParts(values) {
  return values.flatMap((value, index) => [
    ...(index > 0
      ? [index === values.length - 1 ? " and " : ", "]
      : []),
    { bold: true, text: titleCase(value) },
  ]);
}

function joinLabels(labels) {
  if (labels.length < 2) {
    return labels[0] ?? "";
  }
  return `${labels.slice(0, -1).join(", ")} and ${labels.at(-1)}`;
}

function getConstraintParts(thing) {
  const className = thing.constructor.getClassName();
  const values = getThingValues(thing);
  const valueParts = getValueParts(values);
  if (className === "Time") {
    return ["in ", ...valueParts];
  }
  if (REGION_DIMENSIONS.has(className)) {
    return [
      "in ",
      ...valueParts,
      ` ${titleCase(getDimensionLabel(thing))}`,
    ];
  }
  return [
    `where ${titleCase(getDimensionLabel(thing))} is `,
    ...valueParts,
  ];
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

export function getQueryFindingParts(query) {
  const entityClassName = query.entityClass.getClassName();
  const entityLabel =
    ENTITY_LABELS[entityClassName] ?? `${humanizeIdentifier(entityClassName)}s`;
  const aggregateLabel = titleCase(getAggregateLabel(query));
  const groupLabels = query.dimThingList
    .filter((thing) => thing.value === Thing.WILDCARD)
    .map((thing) => titleCase(getDimensionLabel(thing)));
  const constraints = query.dimThingList
    .filter((thing) => thing.value !== Thing.WILDCARD)
    .map(getConstraintParts);
  constraints.push(...(
    ...(query.parentRegionConstraintList ?? [])
      .map(({ parentRegion }) => parentRegion)
      .filter(Boolean)
      .map(getConstraintParts)
  ));

  return [
    `${aggregateLabel} of ${titleCase(entityLabel)}`,
    ...(groupLabels.length > 0 ? [` by ${joinLabels(groupLabels)}`] : []),
    ...(constraints.length > 0
      ? [" ", ...constraints.flatMap((constraint, index) => [
          ...(index > 0 ? [" and "] : []),
          ...constraint,
        ])]
      : []),
  ];
}

export default function getQueryFinding(query) {
  return getQueryFindingParts(query)
    .map((part) => (typeof part === "string" ? part : part.text))
    .join("");
}
