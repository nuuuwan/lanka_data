import Thing from "./thing/Thing.js";
import {
  getConstraintText,
  getDimensionLabel,
  humanizeIdentifier,
  joinLabels,
} from "./QueryFindingLabels.js";

const ENTITY_LABELS = {
  Census: "census records",
  House: "households",
  Person: "population",
  Vote: "votes",
};

const COUNT_LABELS = {
  Person: "population",
  Vote: "valid votes",
};

function getAggregateLabel(query) {
  const entityClassName = query.entityClass.getClassName();
  if (query.aggregate.toLowerCase() === "count") {
    return COUNT_LABELS[entityClassName] ?? "count";
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
  const subject =
    query.aggregate.toLowerCase() === "count" &&
    COUNT_LABELS[entityClassName] !== undefined
      ? aggregateLabel
      : `${aggregateLabel} of ${entityLabel}`;
  const finding = `${subject}${groupedBy}${constrainedBy}`;
  return finding.charAt(0).toUpperCase() + finding.slice(1);
}
