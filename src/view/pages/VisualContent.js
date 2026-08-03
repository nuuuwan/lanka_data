import { useEffect } from "react";

import DataProvenancePanel from "../molecules/DataProvenancePanel.js";
import VisualHeader from "../molecules/VisualHeader.js";
import ChartVisual from "./ChartVisual.js";

export default function VisualContent({
  VisualClass,
  datumSet,
  encodedQuery,
  loadTimeSeconds,
  query,
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
        encodedQuery={encodedQuery}
        datumCount={datumSet.datumList.length}
        datumSet={datumSet}
        loadTimeSeconds={loadTimeSeconds}
      />
      {VisualClass.IS_CHART ? (
        <ChartVisual VisualClass={VisualClass} datumSet={datumSet} />
      ) : (
        <VisualClass datumSet={datumSet} />
      )}
      <DataProvenancePanel provenance={datumSet.provenance} />
    </>
  );
}
