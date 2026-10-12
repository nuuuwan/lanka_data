import { Box, Typography } from "@mui/material";

import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import { VISUAL_LABEL_MIN_FONT_SIZE } from "../../../nonview/constants/ChartLabels.js";

export default function Legend({ items }) {
  return (
    <Box
      sx={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: 1,
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
            maxWidth: "100%",
          }}
        >
          <Box
            sx={{
              width: VISUAL_LABEL_MIN_FONT_SIZE,
              height: VISUAL_LABEL_MIN_FONT_SIZE,
              borderRadius: 0,
              backgroundColor: getMarkColor(item.color),
              flexShrink: 0,
            }}
          />
          <Typography
            variant="caption"
            sx={{
              fontSize: VISUAL_LABEL_MIN_FONT_SIZE,
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
