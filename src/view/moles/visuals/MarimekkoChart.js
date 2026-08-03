import { Box, Typography, useMediaQuery, useTheme } from "@mui/material";
import { ResponsiveMarimekko } from "@nivo/marimekko";

import { FONT_FAMILY } from "../../../AppTheme.js";
import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import InBarLabels from "./InBarLabels.js";
import Legend from "./Legend.js";
import { BarLabelsLayer } from "./MarimekkoLabels.js";
import {
  getColorForDimension,
  getDimensions,
  sortByDominantCategory,
} from "./MarimekkoUtils.js";

export default function MarimekkoChart({ data, xAxisLabel }) {
  const theme = useTheme();
  const screenWidth = useMediaQuery(theme.breakpoints.down("sm")) ? 375 : 1200;
  const sortedData = sortByDominantCategory(data);
  const dimensions = getDimensions(sortedData);
  const color = (id) => getMarkColor(getColorForDimension(id, sortedData));
  return (
    <Box>
      <Box sx={{ height: 400 }}>
        <ResponsiveMarimekko
          data={sortedData}
          id="id"
          value="_barWidth"
          dimensions={dimensions}
          offset="expand"
          layout="vertical"
          theme={{ fontFamily: FONT_FAMILY, text: { fontFamily: FONT_FAMILY } }}
          axisTop={null}
          axisRight={null}
          enableGridY={false}
          axisBottom={{
            orient: "bottom",
            tickSize: 0,
            tickPadding: 0,
            tickRotation: 0,
            legend: xAxisLabel,
            legendOffset: 40,
            legendPosition: "middle",
            format: () => "",
          }}
          layers={[
            "grid",
            "axes",
            "bars",
            (props) => <BarLabelsLayer {...props} screenWidth={screenWidth} />,
            InBarLabels,
          ]}
          axisLeft={{
            orient: "left",
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: "%",
            legendOffset: -50,
            legendPosition: "middle",
            format: (value) => `${Math.round(value * 100)}%`,
          }}
          colors={({ id }) => color(id)}
          borderWidth={1}
          borderColor={{ from: "color", modifiers: [["darker", 0.2]] }}
          margin={{ top: 40, right: 50, bottom: 40, left: 80 }}
          tooltip={({ id, value, color: markColor }) => (
            <Box sx={{ backgroundColor: "white", p: 1, border: 1 }}>
              <Typography variant="body2" sx={{ color: markColor }}>
                {id}: {Math.round(value * 100)}%
              </Typography>
            </Box>
          )}
        />
      </Box>
      <Legend
        items={dimensions.map(({ id }) => ({
          id,
          label: id,
          color: color(id),
        }))}
      />
    </Box>
  );
}

MarimekkoChart.IS_CHART = true;
MarimekkoChart.IS_STACKED = true;
MarimekkoChart.IS_MARIMEKKO = true;
