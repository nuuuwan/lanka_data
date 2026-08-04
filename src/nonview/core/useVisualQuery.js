import { useEffect, useRef, useState } from "react";

import VisualQuery from "./VisualQuery.js";

export default function useVisualQuery(
  isReady,
  visualQueryStr,
  getVisualClass,
) {
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
        nextQuery.visualClass = getVisualClass(nextQuery.visualClassName);
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
  }, [getVisualClass, isReady, visualQueryStr]);
  return {
    errorMessage,
    parseStartTime,
    parseTimeSeconds,
    visualQuery,
  };
}
