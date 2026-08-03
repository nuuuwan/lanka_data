import { useEffect, useRef, useState } from "react";

import VisualQuery from "../../nonview/core/VisualQuery.js";

export default function useVisualQuery(isReady, visualQueryStr) {
  const [visualQuery, setVisualQuery] = useState(null);
  const [parseTimeSeconds, setParseTimeSeconds] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const parseStartTime = useRef(null);
  useEffect(() => {
    if (!isReady) return;
    let cancelled = false;
    async function parse() {
      setVisualQuery(null);
      setParseTimeSeconds(null);
      setErrorMessage(null);
      const startTime = performance.now();
      parseStartTime.current = startTime;
      try {
        const nextQuery = await VisualQuery.fromString(visualQueryStr);
        if (!cancelled) {
          setParseTimeSeconds((performance.now() - startTime) / 1000);
          setVisualQuery(nextQuery);
        }
      } catch (error) {
        console.error("[VisualQueryPage] Could not parse request", error);
        if (!cancelled)
          setErrorMessage(
            error instanceof Error && error.message
              ? error.message
              : "We couldn't understand that request. Please check your choices and try again.",
          );
      }
    }
    parse();
    return () => {
      cancelled = true;
    };
  }, [isReady, visualQueryStr]);
  return {
    errorMessage,
    parseStartTime,
    parseTimeSeconds,
    visualQuery,
  };
}
