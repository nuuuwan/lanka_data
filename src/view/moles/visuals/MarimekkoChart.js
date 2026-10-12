import { Box } from "@mui/material";
import { Marimekko } from "@nivo/marimekko";
import SizedChart from "../../organisms/SizedChart.js";

import { NIVO_THEME } from "../../../AppTheme.js";
import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import {
  CHART_MIN_CATEGORY_WIDTH,
  CHART_HORIZONTAL_MARGIN,
} from "../../../nonview/constants/ChartLabels.js";
import {
  BAR_CHART_MIN_HEIGHT_PX,
  getSlantedXAxis,
  X_AXIS_BOTTOM_MARGIN,
} from "../visual_utils/ChartAxisUtils.js";
import Legend from "./Legend.js";
import MarimekkoPercentageLayer from "../../organisms/MarimekkoPercentageLayer.js";
import {
  getColorForDimension,
  getDimensions,
  sortByDominantCategory,
} from "./MarimekkoUtils.js";

export default function MarimekkoChart({ data, xAxisLabel }) {
  const sortedData = sortByDominantCategory(data);
  const dimensions = getDimensions(sortedData);
  const color = (id) => getMarkColor(getColorForDimension(id, sortedData));
  return (
    <Box>
      <Box
        sx={{
          width: "100%",
          height: BAR_CHART_MIN_HEIGHT_PX,
        }}
      >
        <SizedChart
          chart={Marimekko}
          minWidth={
            sortedData.length * CHART_MIN_CATEGORY_WIDTH +
            CHART_HORIZONTAL_MARGIN
          }
          theme={NIVO_THEME}
          data={sortedData}
          id="id"
          value="_barWidth"
          animate={false}
          dimensions={dimensions}
          offset="expand"
          layout="vertical"
          axisTop={null}
          axisRight={null}
          enableGridY={false}
          axisBottom={getSlantedXAxis(xAxisLabel, {
            format: () => "",
            tickSize: 0,
            tickPadding: 0,
          })}
          layers={["grid", "axes", MarimekkoPercentageLayer]}
          axisLeft={null}
          colors={({ id }) => color(id)}
          borderWidth={1}
          borderColor={{ from: "color", modifiers: [["darker", 0.2]] }}
          margin={{
            top: 40,
            right: 50,
            bottom: X_AXIS_BOTTOM_MARGIN,
            left: 80,
          }}
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
MarimekkoChart.IS_FULL_WIDTH = true;
MarimekkoChart.IS_STACKED = true;
MarimekkoChart.IS_MARIMEKKO = true;
