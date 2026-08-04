import { useMemo, useState } from "react";

import TableResults from "../moles/TableResults.js";
import {
  getTableColumns,
  sortTableData,
} from "../moles/visual_utils/TableData.js";

export default function TableVisual({ datumSet }) {
  const { datumList } = datumSet;
  const [sort, setSort] = useState(null);
  const firstDatum = datumList[0];
  const columns = useMemo(() => getTableColumns(firstDatum), [firstDatum]);
  const sortedData = useMemo(
    () => sortTableData(datumList, columns, sort),
    [columns, datumList, sort],
  );

  function sortBy(columnId) {
    setSort((current) => ({
      columnId,
      direction:
        current?.columnId === columnId && current.direction === "asc"
          ? "desc"
          : "asc",
    }));
  }

  return (
    <TableResults
      columns={columns}
      datumList={sortedData}
      onSort={sortBy}
      sort={sort}
    />
  );
}
