import React from "react";
import { Box, List, Typography } from "@mui/material";
import DatumView from "./DatumView.js";
export default function DatumSetView({ datumSet }) {
  return (
    <Box sx={{ m: 1, p: 1 }}>
      <Typography variant="body1" sx={{ color: "info.main" }}>
        {datumSet.datumList.length} datum(s)
      </Typography>
      {datumSet.datumList.map((datum, index) => (
        <List key={index}>
          <DatumView datum={datum} />
        </List>
      ))}
    </Box>
  );
}
