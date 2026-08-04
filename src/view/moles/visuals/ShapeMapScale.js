import { Typography } from "@mui/material";

import {
  HEX_MAP_SCALE_COLOR,
  HEX_MAP_SCALE_FONT_SIZE,
} from "../../_cons/MapCons.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

export default function ShapeMapScale({ map, shapeName, shapeUnit }) {
  const { shapeValueMin: min, shapeValueMax: max } = map;
  const value =
    max - min < 1
      ? FormatUtils.humanizeValue(min)
      : `${FormatUtils.humanizeValue(min)} to ${FormatUtils.humanizeValue(max)}`;
  return (
    <Typography
      variant="caption"
      sx={{
        color: HEX_MAP_SCALE_COLOR,
        display: "block",
        fontSize: HEX_MAP_SCALE_FONT_SIZE,
        textAlign: "center",
      }}
    >
      1 {shapeName} = {value} {shapeUnit}
    </Typography>
  );
}
