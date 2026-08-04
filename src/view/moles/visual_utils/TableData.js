import FormatUtils from "./FormatUtils.js";

export function getTableColumns(firstDatum) {
  if (!firstDatum) return [];
  const dimensions = firstDatum.query.dimThingList.map((thing, index) => ({
    id: `dimension-${index}`,
    label: thing.constructor.name,
    value: (datum) => FormatUtils.toThingLabel(datum.query.dimThingList[index]),
  }));
  return [
    ...dimensions,
    {
      id: "aggregate",
      label: firstDatum.query.aggregate,
      numeric: true,
      value: (datum) => FormatUtils.humanizeValue(datum.answerThing.value),
      sortValue: (datum) => Number(datum.answerThing.value),
    },
  ];
}

function compareValues(left, right) {
  return String(left).localeCompare(String(right), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

export function sortTableData(datumList, columns, sort) {
  if (!sort) return datumList;
  const column = columns.find(({ id }) => id === sort.columnId);
  return datumList
    .map((datum, index) => ({ datum, index }))
    .sort((left, right) => {
      const getValue = column.sortValue ?? column.value;
      const leftValue = getValue(left.datum);
      const rightValue = getValue(right.datum);
      const comparison = column.numeric
        ? leftValue - rightValue
        : compareValues(leftValue, rightValue);
      return (
        (sort.direction === "asc" ? comparison : -comparison) ||
        left.index - right.index
      );
    })
    .map(({ datum }) => datum);
}
