import { useLayoutEffect, useRef, useState } from "react";

import styles from "./SizedChart.module.css";

export default function SizedChart({ chart: Chart, ...props }) {
  const ref = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const element = ref.current;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize((previous) =>
        previous.width === width && previous.height === height
          ? previous
          : { width, height },
      );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref} className={styles.container}>
      {size.width > 0 && size.height > 0 && (
        <Chart {...props} width={size.width} height={size.height} />
      )}
    </div>
  );
}
