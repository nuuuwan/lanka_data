import { Box, CircularProgress, Typography } from "@mui/material";

export default function LoadingProgress() {
  return (
    <Box
      sx={{
        alignItems: "center",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        py: 4,
      }}
    >
      <CircularProgress
        aria-label="Loading visual"
        color="info"
        size={64}
        thickness={5}
      />
      <Typography color="text.secondary" variant="body2" sx={{ mt: 1 }}>
        Loading visual…
      </Typography>
    </Box>
  );
}
