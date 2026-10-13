import Thing from "./thing/Thing.js";

const ENTITY_LABELS = {
  Census: "census records",
  House: "households",
  Person: "population",
  Vote: "votes",
};

const COUNT_LABELS = {
  Person: "Population",
  Vote: "Votes",
  House: "Households",
};

const DIMENSION_LABELS = {
  DSD: "divisional secretariat division",
  ED: "electoral district",
  GND: "Grama Niladhari division",
  PD: "polling division",
};

const DIMENSION_PLURALS = {
  Country: "Countries",
  District: "Districts",
  DSD: "DSDs",
  ED: "EDs",
  GND: "GNDs",
  PD: "PDs",
  Province: "Provinces",
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

function getRegionName(thing) {
  const ent = thing.getEnt ? thing.getEnt() : null;
  if (ent && ent.name) {
    return ent.name;
  }
  const label = thing.getLabel ? thing.getLabel() : thing.value;
  return label
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function getThingValues(thing) {
  return (thing.valueList ?? [thing.value]).map((value) => {
    const inst = thing.constructor.fromValue(value);
    if (REGION_DIMENSIONS.has(thing.constructor.getClassName())) {
      return getRegionName(inst);
    }
    return inst.getLabel();
  });
}

function joinLabels(labels) {
  if (labels.length < 2) {
    return labels[0] ?? "";
  }
  return `${labels.slice(0, -1).join(", ")} and ${labels.at(-1)}`;
}

function formatRegionNameWithClass(thing) {
  const className = thing.constructor.getClassName();
  const name = joinLabels(getThingValues(thing));
  if (className === "Province") {
    return `the ${name} Province`;
  }
  if (className === "District") {
    return `${name} District`;
  }
  if (className === "ED") {
    return `${name} Electoral District`;
  }
  if (className === "Country") {
    return name;
  }
  return `${name} ${getDimensionLabel(thing)}`;
}

function getSubject(query) {
  const entityClassName = query.entityClass.getClassName();
  if (query.aggregate.toLowerCase() === "count") {
    return COUNT_LABELS[entityClassName] ?? "Count";
  }
  return `${humanizeIdentifier(query.aggregate)} of ${
    ENTITY_LABELS[entityClassName] ?? humanizeIdentifier(entityClassName)
  }`;
}

export default function getQueryFinding(query, visualClassOrName = null) {
  const timeThings = query.dimThingList.filter(
    (t) =>
      t.constructor.getClassName() === "Time" && t.value !== Thing.WILDCARD,
  );
  const timeStr =
    timeThings.length > 0
      ? `for ${joinLabels(timeThings.flatMap(getThingValues))}`
      : "";

  const parentConstraintPhrases = (query.parentRegionConstraintList ?? []).map(
    ({ childClass, parentRegion }) => {
      const childName = childClass.getClassName();
      const childPlural = DIMENSION_PLURALS[childName] ?? `${childName}s`;
      const parentPhrase = formatRegionNameWithClass(parentRegion);
      return `${childPlural} in ${parentPhrase}`;
    },
  );

  const parentConstrainedChildClasses = new Set(
    (query.parentRegionConstraintList ?? []).map(({ childClass }) =>
      childClass.getClassName(),
    ),
  );

  const standaloneConstraints = query.dimThingList
    .filter(
      (t) =>
        t.value !== Thing.WILDCARD && t.constructor.getClassName() !== "Time",
    )
    .map((thing) => {
      const className = thing.constructor.getClassName();
      if (REGION_DIMENSIONS.has(className)) {
        return `in ${formatRegionNameWithClass(thing)}`;
      }
      const values = joinLabels(getThingValues(thing));
      return `where ${getDimensionLabel(thing)} is ${values}`;
    });

  const wildcards = query.dimThingList.filter(
    (t) =>
      t.value === Thing.WILDCARD &&
      !parentConstrainedChildClasses.has(t.constructor.getClassName()),
  );
  const wildcardPhrases = wildcards.map((thing) => {
    const className = thing.constructor.getClassName();
    if (DIMENSION_PLURALS[className]) {
      return DIMENSION_PLURALS[className];
    }
    return getDimensionLabel(thing);
  });

  const subject = getSubject(query);
  const visualName = visualClassOrName
    ? typeof visualClassOrName === "string"
      ? visualClassOrName
      : visualClassOrName.name
    : "";

  const subjectPhrase = visualName ? `${visualName} of ${subject}` : subject;

  const qualifierParts = [
    timeStr,
    ...parentConstraintPhrases,
    ...standaloneConstraints,
    ...wildcardPhrases.map((p) => `by ${p}`),
  ].filter(Boolean);

  if (qualifierParts.length === 0) {
    return subjectPhrase;
  }

  // Combine with proper grammar:
  // e.g. "HexMap of Population for 2024 and PDs in the Uva Province"
  // If first qualifier starts with 'for ' and second with something else, join with ' and '
  let result = subjectPhrase;
  if (timeStr) {
    result += ` ${timeStr}`;
    const otherQualifiers = qualifierParts.slice(1);
    if (otherQualifiers.length > 0) {
      result += ` and ${joinLabels(otherQualifiers)}`;
    }
  } else {
    result += ` ${joinLabels(qualifierParts)}`;
  }

  return result;
}
