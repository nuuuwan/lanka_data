import { useParams } from "react-router-dom";
import { Typography, Box, CircularProgress } from "@mui/material";
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
    <Box>
      <Typography variant="h4" sx={{ mt: 2 }}>
        {queryStr}
      </Typography>
      {lankaData === null ? (
        <CircularProgress sx={{ mt: 2 }} />
      ) : (
        <Typography variant="body1" sx={{ mt: 2 }}>
          {JSON.stringify(lankaData, null, 2)}
        </Typography>
      )}
    </Box>
  );
}
