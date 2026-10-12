import { useRef } from "react";
import { Box, Paper } from "@mui/material";

import useViewportFit from "../../nonview/base/useViewportFit.js";
import {
  VISUAL_PAPER_PADDING_PX,
  VISUAL_VIEWPORT_MAX_RATIO,
} from "../../nonview/constants/APP.js";

export default function FittedVisual({ children }) {
  const containerRef = useRef(null);
  const contentRef = useRef(null);
  const { scale, height, left } = useViewportFit(
    containerRef,
    contentRef,
    VISUAL_VIEWPORT_MAX_RATIO,
    VISUAL_PAPER_PADDING_PX,
  );

  return (
    <Paper
      data-testid="fitted-visual"
      elevation={0}
      sx={{
        bgcolor: "grey.100",
        boxSizing: "border-box",
        maxWidth: `${VISUAL_VIEWPORT_MAX_RATIO * 100}vw`,
        maxHeight: `${VISUAL_VIEWPORT_MAX_RATIO * 100}vh`,
        position: "relative",
        left: "50%",
        transform: "translateX(-50%)",
        overflow: "hidden",
        p: `${VISUAL_PAPER_PADDING_PX}px`,
        width: `${VISUAL_VIEWPORT_MAX_RATIO * 100}vw`,
      }}
    >
      <Box ref={containerRef} sx={{ position: "relative", height }}>
        <Box
          ref={contentRef}
          sx={{
            position: "absolute",
            top: 0,
            left,
            width: "100%",
            display: "flow-root",
            transform: `scale(${scale})`,
            transformOrigin: "top left",
          }}
        >
          {children}
        </Box>
      </Box>
    </Paper>
  );
}
