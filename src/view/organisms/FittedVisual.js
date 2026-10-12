import { useRef, useState } from "react";
import { Box, Paper } from "@mui/material";

import useViewportFit from "../../nonview/base/useViewportFit.js";
import useElementHeight from "../../nonview/base/useElementHeight.js";
import { VisualLayoutContext } from "../../nonview/core/VisualLayoutContext.js";
import {
  VISUAL_PAPER_PADDING_PX,
  VISUAL_VIEWPORT_MAX_RATIO,
} from "../../nonview/constants/APP.js";

export default function FittedVisual({ children, header, footer, visualRef }) {
  const [visualCount, setVisualCount] = useState(1);
  const fitToViewport = visualCount <= 1;
  const headerRef = useRef(null);
  const footerRef = useRef(null);
  const reservedHeight =
    useElementHeight(headerRef) + useElementHeight(footerRef);
  const containerRef = useRef(null);
  const contentRef = useRef(null);
  const { scale, height } = useViewportFit(
    containerRef,
    contentRef,
    VISUAL_VIEWPORT_MAX_RATIO,
    VISUAL_PAPER_PADDING_PX,
    reservedHeight,
    fitToViewport,
  );

  return (
    <VisualLayoutContext.Provider value={setVisualCount}>
      <Box
        ref={visualRef}
        data-testid="fitted-visual-section"
        data-visual-count={visualCount}
        sx={{
          position: "relative",
          left: fitToViewport ? "50%" : undefined,
          transform: fitToViewport ? "translateX(-50%)" : undefined,
          width: fitToViewport
            ? `${VISUAL_VIEWPORT_MAX_RATIO * 100}vw`
            : "100%",
        }}
      >
        <Box ref={headerRef} sx={{ display: "flow-root" }}>
          {header}
        </Box>
        <Paper
          data-testid="fitted-visual"
          elevation={0}
          sx={{
            bgcolor: "common.white",
            boxSizing: "border-box",
            overflow: "hidden",
            p: `${VISUAL_PAPER_PADDING_PX}px`,
            width: "100%",
          }}
        >
          <Box
            ref={containerRef}
            sx={{
              position: "relative",
              height: fitToViewport ? height : "auto",
            }}
          >
            <Box
              ref={contentRef}
              sx={{
                position: fitToViewport ? "absolute" : "relative",
                top: 0,
                left: fitToViewport ? "50%" : 0,
                width: "100%",
                display: "flow-root",
                transform: fitToViewport
                  ? `translateX(-50%) scale(${scale})`
                  : undefined,
                transformOrigin: "top center",
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
    </VisualLayoutContext.Provider>
  );
}
