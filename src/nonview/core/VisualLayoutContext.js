import { createContext, useContext, useLayoutEffect } from "react";

export const VisualLayoutContext = createContext(null);

export function useVisualCount(count) {
  const setVisualCount = useContext(VisualLayoutContext);
  useLayoutEffect(() => {
    if (!setVisualCount) return undefined;
    setVisualCount(count);
    return () => setVisualCount(1);
  }, [count, setVisualCount]);
}
