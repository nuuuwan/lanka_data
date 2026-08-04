import { TableBody, TableCell, TableRow } from "@mui/material";

import styles from "../../organisms/TableVisual.module.css";

export default function TableResultRows({ columns, datumList }) {
  if (datumList.length === 0) {
    return (
      <TableBody>
        <TableRow>
          <TableCell colSpan={Math.max(columns.length, 1)}>
            No query results available.
          </TableCell>
        </TableRow>
      </TableBody>
    );
  }
  return (
    <TableBody>
      {datumList.map((datum, rowIndex) => (
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
      ))}
    </TableBody>
  );
}
