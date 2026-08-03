import { useEffect, useRef, useState } from "react";

import DataSourceFactory from "../../nonview/core/data_source/DataSourceFactory.js";

export default function useVisualData(visualQuery) {
  const [datumSet, setDatumSet] = useState(null);
  const [loadTimeSeconds, setLoadTimeSeconds] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const loadStartTime = useRef(null);
  useEffect(() => {
    setDatumSet(null);
    setLoadTimeSeconds(null);
    setErrorMessage(null);
    if (!visualQuery) return;
    let cancelled = false;
    async function fetchData() {
      const startTime = performance.now();
      loadStartTime.current = startTime;
      try {
        const nextDatumSet = await DataSourceFactory.getDatumSetForQuery(
          visualQuery.query,
        );
        const nextLoadTime = (performance.now() - startTime) / 1000;
        if (!cancelled) {
          setDatumSet(nextDatumSet);
          setLoadTimeSeconds(nextLoadTime);
          setErrorMessage(
            nextDatumSet.datumList.length
              ? null
              : "We couldn't find any data for that request. Please check your choices and try again.",
          );
        }
      } catch (error) {
        console.error("[VisualQueryPage] Could not load requested data", error);
        if (!cancelled)
          setErrorMessage(
            "We couldn't load the data for that request. Please try again.",
          );
      }
    }
    fetchData();
    return () => {
      cancelled = true;
    };
  }, [visualQuery]);
  return { datumSet, errorMessage, loadStartTime, loadTimeSeconds };
}
