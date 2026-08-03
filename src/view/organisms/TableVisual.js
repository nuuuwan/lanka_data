import { useMemo, useState } from "react";
import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
} from "@mui/material";

import FormatUtils from "../moles/visual_utils/FormatUtils.js";
import styles from "./TableVisual.module.css";

function compareValues(left, right) {
  return String(left).localeCompare(String(right), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

export default function TableVisual({ datumSet }) {
  const { datumList } = datumSet;
  const [sort, setSort] = useState(null);
  const firstDatum = datumList[0];
  const dimensionColumns =
    firstDatum?.query.dimThingList.map((thing, index) => ({
      id: `dimension-${index}`,
      label: thing.constructor.name,
      value: (datum) =>
        FormatUtils.toThingLabel(datum.query.dimThingList[index]),
    })) ?? [];
  const columns = firstDatum
    ? [
        ...dimensionColumns,
        {
          id: "aggregate",
          label: firstDatum.query.aggregate,
          numeric: true,
          value: (datum) => FormatUtils.humanizeValue(datum.answerThing.value),
          sortValue: (datum) => Number(datum.answerThing.value),
        },
      ]
    : [];

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

  function sortBy(columnId) {
    setSort((currentSort) => ({
      columnId,
      direction:
        currentSort?.columnId === columnId &&
        currentSort.direction === "asc"
          ? "desc"
          : "asc",
    }));
  }

  return (
    <TableContainer component={Paper} className={styles.container}>
      <Table aria-label="Query results">
        <caption>Query results</caption>
        <TableHead>
          <TableRow>
            {columns.map((column) => (
              <TableCell
                align={column.numeric ? "right" : "left"}
                key={column.id}
                sortDirection={
                  sort?.columnId === column.id ? sort.direction : false
                }
              >
                <TableSortLabel
                  active={sort?.columnId === column.id}
                  direction={
                    sort?.columnId === column.id ? sort.direction : "asc"
                  }
                  onClick={() => sortBy(column.id)}
                >
                  {column.label}
                </TableSortLabel>
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {sortedDatumList.length === 0 ? (
            <TableRow>
              <TableCell colSpan={Math.max(columns.length, 1)}>
                No query results available.
              </TableCell>
            </TableRow>
          ) : (
            sortedDatumList.map((datum, rowIndex) => (
              <TableRow key={rowIndex}>
                {columns.map((column) => (
                  <TableCell
                    align={column.numeric ? "right" : "left"}
                    className={column.numeric ? styles.numericCell : undefined}
                    key={column.id}
                  >
                    {column.value(datum)}
                  </TableCell>
                ))}
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
