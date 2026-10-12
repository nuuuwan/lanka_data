import { useRef } from "react";
import { Box, Paper } from "@mui/material";

import useViewportFit from "../../nonview/base/useViewportFit.js";
import useElementHeight from "../../nonview/base/useElementHeight.js";
import {
  VISUAL_PAPER_PADDING_PX,
  VISUAL_VIEWPORT_MAX_RATIO,
} from "../../nonview/constants/APP.js";

export default function FittedVisual({ children, header, footer }) {
  const headerRef = useRef(null);
  const footerRef = useRef(null);
  const reservedHeight =
    useElementHeight(headerRef) + useElementHeight(footerRef);
  const containerRef = useRef(null);
  const contentRef = useRef(null);
  const { scale, height, left } = useViewportFit(
    containerRef,
    contentRef,
    VISUAL_VIEWPORT_MAX_RATIO,
    VISUAL_PAPER_PADDING_PX,
    reservedHeight,
  );

  return (
    <Box
      data-testid="fitted-visual-section"
      sx={{
        position: "relative",
        left: "50%",
        transform: "translateX(-50%)",
        width: `${VISUAL_VIEWPORT_MAX_RATIO * 100}vw`,
      }}
    >
      <Box ref={headerRef} sx={{ display: "flow-root" }}>
        {header}
      </Box>
      <Paper
        data-testid="fitted-visual"
        elevation={0}
        sx={{
          bgcolor: "grey.100",
          boxSizing: "border-box",
          overflow: "hidden",
          p: `${VISUAL_PAPER_PADDING_PX}px`,
          width: "100%",
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
      <Box ref={footerRef} sx={{ display: "flow-root" }}>
        {footer}
      </Box>
    </Box>
  );
}
