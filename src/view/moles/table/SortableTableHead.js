import { TableCell, TableHead, TableRow, TableSortLabel } from "@mui/material";

export default function SortableTableHead({ columns, onSort, sort }) {
  return (
    <TableHead>
      <TableRow>
        {columns.map((column) => {
          const isActive = sort?.columnId === column.id;
          return (
            <TableCell
              align={column.numeric ? "right" : "left"}
              key={column.id}
              sortDirection={isActive ? sort.direction : false}
            >
              <TableSortLabel
                active={isActive}
                direction={isActive ? sort.direction : "asc"}
                onClick={() => onSort(column.id)}
              >
                {column.label}
              </TableSortLabel>
            </TableCell>
          );
        })}
      </TableRow>
    </TableHead>
  );
}
