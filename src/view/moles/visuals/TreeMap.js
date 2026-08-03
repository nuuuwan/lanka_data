import { ResponsiveTreeMap } from "@nivo/treemap";
import { Box, Typography } from "@mui/material";

import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

export default function TreeMap({ data, xAxisLabel }) {
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
    <Box sx={{ width: "100%", height: "100%", minHeight: 400 }}>
      <ResponsiveTreeMap
        data={treeData}
        identity="id"
        value="value"
        margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
        tile="squarify"
        colors={(node) => getMarkColor(node.data.color)}
        borderColor={{ from: "color", modifiers: [["darker", 0.3]] }}
        label={({ id }) => id}
        labelSkipSize={24}
        labelTextColor={{ from: "color", modifiers: [["darker", 1.8]] }}
        parentLabelPosition="left"
        parentLabelTextColor={{
          from: "color",
          modifiers: [["darker", 2]],
        }}
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
