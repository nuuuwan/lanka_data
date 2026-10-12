import { Pie } from "@nivo/pie";
import SizedChart from "../../organisms/SizedChart.js";
import { Box, Typography } from "@mui/material";

import { NIVO_THEME } from "../../../AppTheme.js";
import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import {
  PIE_CHART_HEIGHT,
  PIE_START_ANGLE,
  PIE_TOTAL_LABEL_OFFSET_PX,
  PIE_TOTAL_LABEL_FONT_SIZE,
  PIE_ARC_LABEL_MIN_ANGLE,
} from "../../../nonview/constants/ChartLabels.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import PieArcLabel from "./PieArcLabel.js";

function getSortedData(data) {
  const categories = new Map();
  for (const datum of data) {
    const existing = categories.get(datum.id);
    categories.set(datum.id, {
      ...datum,
      value: (existing?.value ?? 0) + datum.value,
    });
  }
  return [...categories.values()].sort(
    (first, second) =>
      second.value - first.value || first.id.localeCompare(second.id),
  );
}

export default function PieChart({ data, total, maxTotal }) {
  const scale = maxTotal > 0 && total > 0 ? Math.sqrt(total / maxTotal) : 0;
  const sortedData = getSortedData(data);
  const totalLabel = FormatUtils.humanizeValue(total);

  return (
    <Box sx={{ width: "100%", minHeight: PIE_CHART_HEIGHT * scale }}>
      <Box
        sx={{
          height: PIE_CHART_HEIGHT,
          transform: `scale(${scale})`,
          transformOrigin: "center",
        }}
      >
        <SizedChart
          chart={Pie}
          theme={NIVO_THEME}
          data={sortedData}
          margin={{ top: 40, right: 80, bottom: 40, left: 80 }}
          colors={(arc) => getMarkColor(arc.data.color)}
          borderWidth={1}
          borderColor={{ from: "color", modifiers: [["darker", 0.2]] }}
          arcLabel={({ value }) => FormatUtils.humanizeValue(value)}
          arcLabelsSkipAngle={PIE_ARC_LABEL_MIN_ANGLE}
          arcLabelsComponent={PieArcLabel}
          enableArcLinkLabels={false}
          innerRadius={0.5}
          startAngle={PIE_START_ANGLE}
          layers={[
            "arcs",
            "arcLabels",
            ({ centerX, centerY }) => (
              <text
                x={centerX}
                y={centerY + PIE_TOTAL_LABEL_OFFSET_PX}
                textAnchor="middle"
                style={{ fontSize: PIE_TOTAL_LABEL_FONT_SIZE, fontWeight: 600 }}
              >
                {totalLabel}
              </text>
            ),
          ]}
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
PieChart.IS_PIE = true;
