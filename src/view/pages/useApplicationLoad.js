import { useEffect, useRef, useState } from "react";

export default function useApplicationLoad(isReady) {
  const startTime = useRef(performance.now());
  const [loadTimeSeconds, setLoadTimeSeconds] = useState(isReady ? 0 : null);
  useEffect(() => {
    if (isReady && loadTimeSeconds === null) {
      setLoadTimeSeconds((performance.now() - startTime.current) / 1000);
    }
  }, [isReady, loadTimeSeconds]);
  return { startTime, loadTimeSeconds };
}
