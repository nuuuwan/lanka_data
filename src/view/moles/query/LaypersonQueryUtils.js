import CategoryConcept from "../../../nonview/core/thing/concept/category_concept/CategoryConcept.js";
import ThingFactory from "../../../nonview/core/thing/thing_factory/ThingFactory.js";

export function getVisualQueryParts(value) {
  const [entity = "", dimensions = "", aggregate = "", visual = ""] =
    value.split("/");
  return { entity, dimensions, aggregate, visual };
}

export function getVisualLabel(value) {
  return value.replaceAll("_", " ").replace(/([a-z])([A-Z])/g, "$1 $2");
}

export function getDimensionParts(value) {
  return value.split("+").map((dimension) => {
    const operatorIndex = dimension.search(/[=:<]/);
    return operatorIndex === -1
      ? { field: dimension, operator: "", value: "" }
      : {
          field: dimension.slice(0, operatorIndex),
          operator: dimension[operatorIndex],
          value: dimension.slice(operatorIndex + 1),
        };
  });
}

export function getDimensionString({ field, operator, value }) {
  return `${field}${operator}${operator ? value : ""}`;
}

export function getValueOptions(field) {
  try {
    const ThingClass = ThingFactory.fromKey(field);
    if (!(ThingClass.prototype instanceof CategoryConcept)) return null;
    return ThingClass.validValues().map((value) => ({
      value,
      label: value,
      color: ThingClass.fromValue(value).getColor(),
    }));
  } catch {
    return null;
  }
}
