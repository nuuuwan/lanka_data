import { ResponsivePie } from "@nivo/pie";
import { Box, Typography } from "@mui/material";

import { FONT_FAMILY } from "../../../AppTheme.js";
import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

const CHART_HEIGHT = 400;

export default function PieChart({ data, total, maxTotal }) {
  const scale = maxTotal > 0 && total > 0 ? Math.sqrt(total / maxTotal) : 0;

  return (
    <Box sx={{ width: "100%", height: CHART_HEIGHT * scale }}>
      <Box
        sx={{
          height: CHART_HEIGHT,
          transform: `scale(${scale})`,
          transformOrigin: "center",
        }}
      >
        <ResponsivePie
          data={data}
          margin={{ top: 40, right: 80, bottom: 40, left: 80 }}
          theme={{ fontFamily: FONT_FAMILY }}
          colors={(arc) => getMarkColor(arc.data.color)}
          borderWidth={1}
          borderColor={{ from: "color", modifiers: [["darker", 0.2]] }}
          arcLabel={({ value }) => FormatUtils.humanizeValue(value)}
          arcLabelsSkipAngle={10}
          arcLabelsTextColor={{ from: "color", modifiers: [["darker", 2]] }}
          tooltip={({ datum }) => (
            <Typography variant="body2">
              {datum.id}: {FormatUtils.humanizeValue(datum.value)}
            </Typography>
          )}
          role="img"
          ariaLabel="Pie chart"
        />
      </Box>
    </Box>
  );
}

PieChart.IS_CHART = true;
