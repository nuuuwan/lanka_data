import { ResponsiveAreaBump } from "@nivo/bump";
import { Box, Typography } from "@mui/material";

import { FONT_FAMILY } from "../../../AppTheme.js";
import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

function getSeriesKeys(data) {
  if (data.length === 0) {
    return [];
  }
  return Object.keys(data[0]).filter(
    (key) => key !== "id" && key !== "_barWidth" && !key.endsWith("Color"),
  );
}

function getMixingScore(left, right, seriesKeys) {
  let score = 0;

  for (let i = 0; i < seriesKeys.length; i++) {
    for (let j = i + 1; j < seriesKeys.length; j++) {
      const firstKey = seriesKeys[i];
      const secondKey = seriesKeys[j];
      const leftOrder = Math.sign(
        (left[firstKey] ?? 0) - (left[secondKey] ?? 0),
      );
      const rightOrder = Math.sign(
        (right[firstKey] ?? 0) - (right[secondKey] ?? 0),
      );

      if (leftOrder !== 0 && rightOrder !== 0 && leftOrder !== rightOrder) {
        score++;
      }
    }
  }

  return score;
}

function getPathMixingScore(path, seriesKeys) {
  return path
    .slice(1)
    .reduce(
      (score, row, index) =>
        score + getMixingScore(path[index], row, seriesKeys),
      0,
    );
}

export function sortAreaBumpXAxis(data) {
  if (data.length < 2) {
    return [...data];
  }

  const seriesKeys = getSeriesKeys(data);
  let bestPath = [...data];
  let bestScore = getPathMixingScore(bestPath, seriesKeys);

  for (let startIndex = 0; startIndex < data.length; startIndex++) {
    const path = [data[startIndex]];
    const remaining = data
      .map((row, index) => ({ row, index }))
      .filter(({ index }) => index !== startIndex);

    while (remaining.length > 0) {
      const previous = path.at(-1);
      let nearestIndex = 0;
      let nearestScore = getMixingScore(previous, remaining[0].row, seriesKeys);

      for (let i = 1; i < remaining.length; i++) {
        const score = getMixingScore(previous, remaining[i].row, seriesKeys);
        if (score < nearestScore) {
          nearestIndex = i;
          nearestScore = score;
        }
      }

      path.push(remaining.splice(nearestIndex, 1)[0].row);
    }

    const score = getPathMixingScore(path, seriesKeys);
    if (score < bestScore) {
      bestPath = path;
      bestScore = score;
    }
  }

  return bestPath;
}

export function toAreaBumpData(data) {
  const sortedData = sortAreaBumpXAxis(data);
  return getSeriesKeys(sortedData).map((key) => ({
    id: key,
    color: sortedData.find((row) => row[`${key}Color`])?.[`${key}Color`],
    data: sortedData.map((row) => ({
      x: row.id,
      y: row[key] ?? 0,
    })),
  }));
}

export default function AreaBump({ data, xAxisLabel }) {
  const series = toAreaBumpData(data);

  return (
    <Box sx={{ width: "100%", height: "100%", minHeight: 400 }}>
      <ResponsiveAreaBump
        data={series}
        margin={{ top: 40, right: 100, bottom: 60, left: 100 }}
        spacing={8}
        colors={(serie) => getMarkColor(serie.color)}
        blendMode="multiply"
        theme={{ fontFamily: FONT_FAMILY }}
        startLabel
        endLabel
        axisTop={null}
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: -45,
          legend: xAxisLabel,
          legendPosition: "middle",
          legendOffset: 50,
        }}
        tooltip={({ serie }) => (
          <Typography variant="body2">
            {serie.id}:{" "}
            {FormatUtils.humanizeValue(
              serie.points.reduce((sum, point) => sum + point.data.y, 0),
            )}
          </Typography>
        )}
        role="img"
      />
    </Box>
  );
}

AreaBump.IS_CHART = true;
AreaBump.IS_STACKED = true;
