import { Axis } from "@nivo/axes";
import { useTooltip } from "@nivo/tooltip";

import {
  PERCENTAGE_AXIS_TICKS,
  PERCENTAGE_AXIS_LEGEND_OFFSET,
} from "../../nonview/constants/ChartLabels.js";
import InBarLabels from "../moles/visuals/InBarLabels.js";
import { BarLabelsLayer } from "../moles/visuals/MarimekkoLabels.js";
import MarimekkoTooltip from "../moles/visuals/MarimekkoTooltip.js";

export default function MarimekkoPercentageLayer({
  bars,
  data,
  dimensionsScale,
}) {
  const { showTooltipFromEvent, hideTooltip } = useTooltip();
  const scale = dimensionsScale.copy().domain([0, 1]);
  const height = scale(0) - scale(1);
  const correctedBars = bars.map((bar) => {
    const lower = dimensionsScale.invert(bar.y + bar.height);
    const upper = dimensionsScale.invert(bar.y);
    return {
      ...bar,
      y: scale(upper),
      height: scale(lower) - scale(upper),
      percentage: upper - lower,
    };
  });

  return (
    <>
      <Axis
        axis="y"
        scale={scale}
        length={height}
        ticksPosition="before"
        tickValues={PERCENTAGE_AXIS_TICKS}
        tickSize={5}
        tickPadding={5}
        legend="%"
        legendOffset={PERCENTAGE_AXIS_LEGEND_OFFSET}
        legendPosition="middle"
        format={(value) => `${Math.round(value * 100)}%`}
      />
      {correctedBars.map((bar) => {
        const showTooltip = (event) =>
          showTooltipFromEvent(
            <MarimekkoTooltip
              id={bar.dimension.id}
              value={bar.percentage}
              color={bar.color}
            />,
            event,
          );
        return (
          <rect
            key={bar.key}
            x={bar.x}
            y={bar.y}
            width={bar.width}
            height={bar.height}
            fill={bar.color}
            stroke={bar.borderColor}
            strokeWidth={bar.borderWidth}
            onMouseEnter={showTooltip}
            onMouseMove={showTooltip}
            onMouseLeave={hideTooltip}
          />
        );
      })}
      <BarLabelsLayer data={data} />
      <InBarLabels bars={correctedBars} />
    </>
  );
}
