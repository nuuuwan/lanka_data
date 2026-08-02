import { ResponsiveTreeMap } from "@nivo/treemap";
import { Box, Typography, useTheme } from "@mui/material";

import { FONT_FAMILY } from "../../../AppTheme.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

export default function TreeMap({ data, xAxisLabel }) {
  const theme = useTheme();
  const children = Array.from(
    data
      .reduce((nodesById, node) => {
        const existingNode = nodesById.get(node.id);
        nodesById.set(
          node.id,
          existingNode
            ? { ...existingNode, value: existingNode.value + node.value }
            : node,
        );
        return nodesById;
      }, new Map())
      .values(),
  );
  const treeData = {
    id: xAxisLabel || "Data",
    children,
  };

  return (
    <Box sx={{ height: 400 }}>
      <ResponsiveTreeMap
        data={treeData}
        identity="id"
        value="value"
        margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
        tile="squarify"
        colors={(node) => node.data.color ?? theme.palette.primary.main}
        borderColor={{ from: "color", modifiers: [["darker", 0.3]] }}
        label={({ id }) => id}
        labelSkipSize={24}
        labelTextColor={{ from: "color", modifiers: [["darker", 1.8]] }}
        parentLabelPosition="left"
        parentLabelTextColor={{
          from: "color",
          modifiers: [["darker", 2]],
        }}
        theme={{ fontFamily: FONT_FAMILY }}
        tooltip={({ node }) => (
          <Typography variant="body2">
            {node.id}: {FormatUtils.humanizeValue(node.value)}
          </Typography>
        )}
        role="img"
        ariaLabel="Tree map"
      />
    </Box>
  );
}

TreeMap.IS_CHART = true;
