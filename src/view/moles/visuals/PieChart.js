import { ResponsivePie } from "@nivo/pie";
import { Box, Typography } from "@mui/material";

import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

export default function PieChart({ data }) {
  return (
    <Box sx={{ height: 400 }}>
      <ResponsivePie
        data={data}
        margin={{ top: 40, right: 80, bottom: 40, left: 80 }}
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
  );
}

PieChart.IS_CHART = true;
