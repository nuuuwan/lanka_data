import Thing from "./thing/Thing.js";
import {
  COUNT_LABELS,
  ENTITY_LABELS,
  getAggregateLabel,
  getConstraintText,
  getDimensionLabel,
  humanizeIdentifier,
  joinLabels,
} from "./QueryFindingUtils.js";

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
