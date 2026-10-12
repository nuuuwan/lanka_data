import { useLayoutEffect, useState } from "react";

export default function useViewportFit(
  containerRef,
  contentRef,
  maxViewportRatio,
  paddingPx,
  reservedHeight = 0,
  enabled = true,
  minScale = 0,
) {
  const [fit, setFit] = useState({ scale: 1, height: 0 });

  useLayoutEffect(() => {
    if (!enabled) return undefined;
    const container = containerRef.current;
    const content = contentRef.current;
    let contentHeight = content.offsetHeight;
    const update = () => {
      const width = container.clientWidth;
      const contentWidth = Math.max(width, content.scrollWidth);
      const availableHeight = Math.max(
        0,
        window.innerHeight * maxViewportRatio - paddingPx * 2 - reservedHeight,
      );
      const scale = Math.min(
        1,
        contentWidth > 0 ? width / contentWidth : 1,
        contentHeight > 0
          ? Math.max(minScale, availableHeight / contentHeight)
          : 1,
      );
      const next = {
        scale,
        height: contentHeight * scale,
      };
      setFit((previous) =>
        previous.scale === next.scale && previous.height === next.height
          ? previous
          : next,
      );
    };
    const observer = new ResizeObserver((entries) => {
      const contentEntry = entries.find((entry) => entry.target === content);
      if (contentEntry) {
        contentHeight = contentEntry.borderBoxSize[0].blockSize;
      }
      update();
    });
    observer.observe(container);
    observer.observe(content);
    window.addEventListener("resize", update);
    update();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [
    containerRef,
    contentRef,
    maxViewportRatio,
    paddingPx,
    reservedHeight,
    enabled,
    minScale,
  ]);

  return fit;
}
