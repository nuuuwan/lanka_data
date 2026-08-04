import { Box, CircularProgress, Typography } from "@mui/material";

export default function LoadingProgress({
  ariaLabel = "Loading visual",
  label = "Loading visual…",
}) {
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
        aria-label={ariaLabel}
        color="info"
        size={64}
        thickness={5}
      />
      <Typography color="text.secondary" variant="body2" sx={{ mt: 1 }}>
        {label}
      </Typography>
    </Box>
  );
}
