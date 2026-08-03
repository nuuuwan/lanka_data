import { Box, CircularProgress } from "@mui/material";

export default function LoadingProgress() {
  return (
    <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
      <CircularProgress
        aria-label="Loading visual"
        color="info"
        size={64}
        thickness={5}
      />
    </Box>
  );
}
