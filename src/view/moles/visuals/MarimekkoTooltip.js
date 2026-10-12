import { Box, Typography } from "@mui/material";

export default function MarimekkoTooltip({ id, value, color }) {
  return (
    <Box sx={{ backgroundColor: "white", p: 1, border: 1 }}>
      <Typography variant="body2" sx={{ color }}>
        {id}: {Math.round(value * 100)}%
      </Typography>
    </Box>
  );
}
