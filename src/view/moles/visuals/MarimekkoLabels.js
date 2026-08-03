import { FONT_FAMILY } from "../../../AppTheme.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

const getFontScale = (screenWidth) => screenWidth / 1200;

export function BarLabelsLayer({ data, screenWidth }) {
  const congestionScale = Math.max(0.7, 1 / Math.sqrt(data.length));
  const fontScale = getFontScale(screenWidth) * congestionScale;
  return (
    <>
      {data.map((datum) =>
        datum.width < 20 ? null : (
          <text
            key={datum.id}
            x={datum.x + datum.width / 2}
            y={datum.y + datum.height + 12}
            textAnchor="middle"
            dominantBaseline="hanging"
            style={{
              fontFamily: FONT_FAMILY,
              fontSize: Math.max(
                6,
                Math.min(10, (datum.width / 10) * fontScale),
              ),
              fill: "#333",
            }}
          >
            {datum.id}
          </text>
        ),
      )}
    </>
  );
}

function isLightColor(color) {
  const hex = color.replace("#", "");
  const channels = [0, 2, 4].map(
    (start) => parseInt(hex.substring(start, start + 2), 16) / 255,
  );
  return 0.299 * channels[0] + 0.587 * channels[1] + 0.114 * channels[2] > 0.5;
}

export function CellLabelsLayer({ bars, screenWidth }) {
  return (
    <>
      {bars.map((bar) => {
        const { x, y, width, height } = bar;
        if (width < 24 || height < 16) return null;
        const light = isLightColor(bar.color);
        const value = bar.datum.data[bar.dimension.id] ?? bar.value;
        return (
          <text
            key={bar.key}
            x={x + width / 2}
            y={y + height / 2}
            textAnchor="middle"
            dominantBaseline="middle"
            style={{
              fontFamily: FONT_FAMILY,
              fontSize: Math.max(
                7,
                Math.min(10, 10 * getFontScale(screenWidth)),
              ),
              fill: light ? "black" : "white",
              stroke: light ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
              strokeWidth: 0.3,
            }}
          >
            {FormatUtils.humanizeValue(value)}
          </text>
        );
      })}
    </>
  );
}
