import { Box } from "@mui/material";

import BasicDimensionValue from "./BasicDimensionValue.js";
import ParentDimensionValue from "./ParentDimensionValue.js";

export default function LaypersonDimensionValue(props) {
  const { dimension } = props;
  if (!dimension.operator) {
    return <Box sx={{ display: { xs: "none", sm: "block" } }} />;
  }
  if (dimension.operator === "<") {
    return <ParentDimensionValue {...props} />;
  }
  return <BasicDimensionValue {...props} />;
}
