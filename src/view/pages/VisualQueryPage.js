import { useParams } from "react-router-dom";
import { Typography, Box, CircularProgress } from "@mui/material";
import { useState, useEffect } from "react";
import Census2024 from "../../nonview/core/Census2024.js";
import DatumSetView from "../moles/DatumSetView.js";
import Query from "../../nonview/core/Query.js";
import VisualQuery from "../../nonview/core/VisualQuery.js";

export default function VisualQueryPage() {
  const { "*": visualQueryStr } = useParams();

  const [datumSet, setDatumSet] = useState(null);
  useEffect(() => {
    async function fetch() {
      const visualQuery = VisualQuery.fromString(visualQueryStr);
      setDatumSet(await Census2024.getDatumSetForQuery(visualQuery.query));
    }
    fetch();
  }, [visualQueryStr]);

  return (
    <Box sx={{ m: 1, p: 1 }}>
      <Typography variant="h6" sx={{ mt: 2, color: "info.main" }}>
        Lanka Data
      </Typography>
      <Typography variant="h4" sx={{ mt: 2 }}>
        {visualQueryStr}
      </Typography>
      {datumSet === null ? (
        <CircularProgress sx={{ m: 2 }} />
      ) : (
        <DatumSetView datumSet={datumSet} />
      )}
    </Box>
  );
}
