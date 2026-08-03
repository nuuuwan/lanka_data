import { useEffect } from "react";
import { Typography } from "@mui/material";

import FormatUtils from "../moles/visual_utils/FormatUtils.js";
import DataProvenancePanel from "../molecules/DataProvenancePanel.js";
import VisualHeading from "../molecules/VisualHeading.js";
import ChartVisual from "./ChartVisual.js";

export default function VisualContent({
  VisualClass,
  datumSet,
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
      <VisualHeading query={query} datumSet={datumSet} />
      <Typography
        data-testid="datums-count"
        variant="caption"
        sx={{ color: "text.secondary" }}
      >
        {datumSet.datumList.length} datums loaded in{" "}
        {FormatUtils.humanizeDuration(loadTimeSeconds)}
      </Typography>
      {VisualClass.IS_CHART ? (
        <ChartVisual VisualClass={VisualClass} datumSet={datumSet} />
      ) : (
        <VisualClass datumSet={datumSet} />
      )}
      <DataProvenancePanel provenance={datumSet.provenance} />
    </>
  );
}
