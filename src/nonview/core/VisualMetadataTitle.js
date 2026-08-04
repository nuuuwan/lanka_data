import { getTitlePopulationLabel, humanize } from "./VisualMetadataLabels.js";

export function getUnitsLabel(query, datumSet, populationLabel) {
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

export function getTitleLabel(query) {
  const entityName = query.entityClass.getClassName();
  if (query.aggregate.toLowerCase() === "count") {
    const countLabel = { Person: "Population", Vote: "Valid votes" }[
      entityName
    ];
    if (countLabel) {
      return countLabel;
    }
  }
  const measure = humanize(query.aggregate);
  const titlePopulation = getTitlePopulationLabel(query);
  return `${measure.charAt(0).toUpperCase()}${measure.slice(1)} of ${titlePopulation}`;
}
