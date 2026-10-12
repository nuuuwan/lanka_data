import { useEffect } from "react";

import DataProvenancePanel from "../molecules/DataProvenancePanel.js";
import VisualHeader from "../molecules/VisualHeader.js";
import FittedVisual from "../organisms/FittedVisual.js";
import ChartVisual from "./ChartVisual.js";

export default function VisualContent({
  VisualClass,
  datumSet,
  loadTimeSeconds,
  query,
  visualTitleRef,
}) {
  useEffect(() => {
    console.debug(
      `[VisualQueryPage] Displaying ${VisualClass.name} with ${datumSet.datumList.length} datums`,
    );
  }, [VisualClass, datumSet]);
  return (
    <>
      <VisualHeader
        query={query}
        datumCount={datumSet.datumList.length}
        loadTimeSeconds={loadTimeSeconds}
        titleRef={visualTitleRef}
      />
      <FittedVisual>
        {VisualClass.IS_CHART ? (
          <ChartVisual VisualClass={VisualClass} datumSet={datumSet} />
        ) : (
          <VisualClass datumSet={datumSet} />
        )}
      </FittedVisual>
      <DataProvenancePanel provenance={datumSet.provenance} />
    </>
  );
}
