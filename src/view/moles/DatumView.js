import React from "react";
import { Typography, Chip, ListItem } from "@mui/material";

export default function DatumView({ datum }) {
  return (
    <ListItem>
      {datum.keyList.map((key, index) => (
        <Chip key={index} label={key} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
      ))}

      <Typography variant="body2" sx={{ ml: 1, color: "text.secondary" }}>
        {datum.value}
      </Typography>
    </ListItem>
  );
}
