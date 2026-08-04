import DataProvenancePanel from "../moles/DataProvenancePanel.js";
import VisualHeader from "../moles/VisualHeader.js";
import ChartVisual from "./ChartVisual.js";

export default function VisualContent({
  VisualClass,
  datumSet,
  loadTimeSeconds,
  query,
  visualTitleRef,
}) {
  return (
    <>
      <VisualHeader
        query={query}
        datumCount={datumSet.datumList.length}
        loadTimeSeconds={loadTimeSeconds}
        titleRef={visualTitleRef}
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
