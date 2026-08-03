import { Box, Typography, useMediaQuery, useTheme } from "@mui/material";

import { getMarkColor } from "../../../nonview/constants/COLORS.js";

export default function Legend({ items }) {
  const theme = useTheme();
  const isSmall = useMediaQuery(theme.breakpoints.down("sm"));

  return (
    <Box
      sx={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: isSmall ? 0.5 : 1,
        mt: 2,
        mb: 1,
        px: 1,
      }}
    >
      {items.map((item) => (
        <Box
          key={item.id}
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.5,
            minWidth: "fit-content",
            maxWidth: isSmall ? "100%" : "calc(50% - 8px)",
          }}
        >
          <Box
            sx={{
              width: isSmall ? 10 : 12,
              height: isSmall ? 10 : 12,
              borderRadius: 0,
              backgroundColor: getMarkColor(item.color),
              flexShrink: 0,
            }}
          />
          <Typography
            variant="caption"
            sx={{
              fontSize: isSmall ? "0.65rem" : "0.75rem",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
            title={item.label}
          >
            {item.label}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}
