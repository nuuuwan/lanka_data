import { Paper, Table, TableContainer } from "@mui/material";
import { useMemo, useState } from "react";

import FormatUtils from "../../nonview/core/visual/FormatUtils.js";
import SortableTableHead from "../moles/table/SortableTableHead.js";
import TableResultRows from "../moles/table/TableResultRows.js";
import styles from "./TableVisual.module.css";

function compareValues(left, right) {
  return String(left).localeCompare(String(right), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

function getColumns(firstDatum) {
  if (!firstDatum) {
    return [];
  }
  const dimensionColumns = firstDatum.query.dimThingList.map(
    (thing, index) => ({
      id: `dimension-${index}`,
      label: thing.constructor.name,
      value: (datum) =>
        FormatUtils.toThingLabel(datum.query.dimThingList[index]),
    }),
  );
  return [
    ...dimensionColumns,
    {
      id: "aggregate",
      label: firstDatum.query.aggregate,
      numeric: true,
      value: (datum) => FormatUtils.humanizeValue(datum.answerThing.value),
      sortValue: (datum) => Number(datum.answerThing.value),
    },
  ];
}

export default function TableVisual({ datumSet }) {
  const { datumList } = datumSet;
  const [sort, setSort] = useState(null);
  const columns = useMemo(() => getColumns(datumList[0]), [datumList]);
  const sortedDatumList = useMemo(() => {
    if (!sort) {
      return datumList;
    }
    const column = columns.find(({ id }) => id === sort.columnId);
    return datumList
      .map((datum, index) => ({ datum, index }))
      .sort((left, right) => {
        const leftValue = (column.sortValue ?? column.value)(left.datum);
        const rightValue = (column.sortValue ?? column.value)(right.datum);
        const comparison = column.numeric
          ? leftValue - rightValue
          : compareValues(leftValue, rightValue);
        return (
          (sort.direction === "asc" ? comparison : -comparison) ||
          left.index - right.index
        );
      })
      .map(({ datum }) => datum);
  }, [columns, datumList, sort]);
  const sortBy = (columnId) =>
    setSort((currentSort) => ({
      columnId,
      direction:
        currentSort?.columnId === columnId && currentSort.direction === "asc"
          ? "desc"
          : "asc",
    }));

  return (
    <TableContainer component={Paper} className={styles.container}>
      <Table aria-label="Query results">
        <caption>Query results</caption>
        <SortableTableHead columns={columns} onSort={sortBy} sort={sort} />
        <TableResultRows columns={columns} datumList={sortedDatumList} />
      </Table>
    </TableContainer>
  );
}
