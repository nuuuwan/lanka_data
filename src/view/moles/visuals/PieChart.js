import { ResponsivePie } from "@nivo/pie";
import { Box, Typography } from "@mui/material";

import { FONT_FAMILY } from "../../../AppTheme.js";
import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import Legend from "./Legend.js";

export default function PieChart({ data }) {
  return (
    <Box>
      <Box sx={{ height: 400 }}>
        <ResponsivePie
          data={data}
          margin={{ top: 40, right: 80, bottom: 40, left: 80 }}
          theme={{ fontFamily: FONT_FAMILY }}
          colors={(arc) => getMarkColor(arc.data.color)}
          borderWidth={1}
          borderColor={{ from: "color", modifiers: [["darker", 0.2]] }}
          innerRadius={0.5}
          tooltip={({ datum }) => (
            <Typography variant="body2">
              {datum.id}: {FormatUtils.humanizeValue(datum.value)}
            </Typography>
          )}
          role="img"
          ariaLabel="Pie chart"
        />
      </Box>
      <Legend
        items={data.map((item) => ({
          id: item.id,
          label: item.id,
          color: item.color,
        }))}
      />
    </Box>
  );
}

PieChart.IS_CHART = true;
