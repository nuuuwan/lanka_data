import { Box } from "@mui/material";

export default function SQLText({ children }) {
  return (
    <Box component="span" sx={{ color: "primary.light" }}>
      {children}
    </Box>
  );
}
