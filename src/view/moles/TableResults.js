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

import styles from "./TableResults.module.css";

export default function TableResults({ columns, datumList, onSort, sort }) {
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
                  onClick={() => onSort(column.id)}
                >
                  {column.label}
                </TableSortLabel>
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {datumList.length ? (
            datumList.map((datum, rowIndex) => (
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
          ) : (
            <TableRow>
              <TableCell colSpan={Math.max(columns.length, 1)}>
                No query results available.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
