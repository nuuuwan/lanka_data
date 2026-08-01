import React from "react";
import { Box, Typography, List } from "@mui/material";
import DatumView from "./DatumView.js";
export default function DatumSetView({ datumSet }) {
  return (
    <Box>
      {datumSet.datumList.map((datum, index) => (
        <List key={index}>
          <DatumView datum={datum} />
        </List>
      ))}
    </Box>
  );
}
