import { useEffect, useState } from "react";

function elapsed(startTime, currentTime) {
  return startTime.current === null
    ? 0
    : Math.max(0, currentTime - startTime.current) / 1000;
}

export default function useLoadingSteps({
  application,
  datumSet,
  generationStartTime,
  isLoading,
  isReady,
  isVisualReady,
  load,
  parse,
  visualDataReady,
  VisualClass,
}) {
  const [currentTime, setCurrentTime] = useState(() => performance.now());
  useEffect(() => {
    if (!isLoading) return undefined;
    let frameId;
    const update = () => {
      setCurrentTime(performance.now());
      frameId = requestAnimationFrame(update);
    };
    frameId = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frameId);
  }, [isLoading]);
  return [
    {
      label: "Loading application data",
      status: isReady ? "complete" : "active",
      durationSeconds:
        application.loadTimeSeconds ??
        elapsed(application.startTime, currentTime),
    },
    {
      label: "Understanding request",
      status: !isReady ? "pending" : VisualClass ? "complete" : "active",
      durationSeconds:
        parse.parseTimeSeconds ?? elapsed(parse.parseStartTime, currentTime),
    },
    {
      label: "Loading visual data",
      status: !VisualClass
        ? "pending"
        : datumSet === null || load.loadTimeSeconds === null
          ? "active"
          : "complete",
      durationSeconds:
        load.loadTimeSeconds ?? elapsed(load.loadStartTime, currentTime),
    },
    {
      label: "Generating visual",
      status: !visualDataReady
        ? "pending"
        : isVisualReady
          ? "complete"
          : "active",
      durationSeconds: elapsed(generationStartTime, currentTime),
    },
  ];
}
