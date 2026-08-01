import { useParams } from "react-router-dom";
import { Typography, Box, Paper, CircularProgress } from "@mui/material";
import { useState, useEffect } from "react";
import Census2024 from "../../nonview/core/Census2024.js";
export default function QueryPage() {
  const { "*": queryStr } = useParams();

  const [lankaData, setLankaData] = useState(null);
  useEffect(() => {
    async function fetch() {
      const lankaData = await Census2024.getLankaDataForQuery(queryStr);
      setLankaData(lankaData);
    }
    fetch();
  }, [queryStr]);

  return (
    <Box sx={{ m: 1, p: 1 }}>
      <Typography variant="h6" sx={{ mt: 2, color: "primary.light" }}>
        Lanka Data
      </Typography>
      <Typography variant="h4" sx={{ mt: 2 }}>
        {queryStr}
      </Typography>
      {lankaData === null ? (
        <CircularProgress sx={{ m: 2 }} />
      ) : (
        <Paper sx={{ m: 1, p: 1 }}>
          <Typography variant="body1" sx={{ mt: 2 }}>
            <pre>{JSON.stringify(lankaData, null, 2)}</pre>
          </Typography>
        </Paper>
      )}
    </Box>
  );
}
