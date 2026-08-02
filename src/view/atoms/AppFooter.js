import { Box, Link, Typography } from "@mui/material";
import GitHubIcon from "@mui/icons-material/GitHub";

import VERSION from "../../nonview/cons/VERSION.js";

export default function AppFooter() {
  return (
    <Box sx={{ m: 2, textAlign: "center" }}>
      <Typography
        variant="caption"
        sx={{
          color: "info.main",
          display: "inline-flex",
          alignItems: "center",
          gap: 0.5,
        }}
      >
        v{VERSION.DATETIME_STR} by{" "}
        <Link
          href="https://github.com/nuuuwan/lanka_data"
          target="_blank"
          rel="noopener noreferrer"
          sx={{
            color: "info.main",
            display: "inline-flex",
            alignItems: "center",
            gap: 0.25,
          }}
        >
          <GitHubIcon sx={{ fontSize: "0.875rem" }} />
          @nuuuwan
        </Link>
      </Typography>
    </Box>
  );
}
