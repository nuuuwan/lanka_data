import { useParams } from "react-router-dom";
import { Typography, Box, CircularProgress } from "@mui/material";
import { useState, useEffect, useMemo } from "react";
import Census2024 from "../../nonview/core/Census2024.js";
import VisualQuery from "../../nonview/core/VisualQuery.js";
import ChartDataUtils from "../moles/visual_utils/ChartDataUtils.js";
import DimensionUtils from "../moles/visual_utils/DimensionUtils.js";
import MultiChartLayout from "../moles/visual_utils/MultiChartLayout.js";

function useChartFacets(datumSet) {
  const { datumList } = datumSet;
  const xAxisDimIndex = DimensionUtils.getXAxisDimIndex(datumList);
  const facetDimIndexes = DimensionUtils.getFacetDimIndexes(datumList);
  const xAxisDimName = DimensionUtils.getDimName(datumList, xAxisDimIndex);
  const yAxisLabel = datumList[0]?.query.aggregate ?? "";
  const facets = ChartDataUtils.groupDataByFacet(
    datumList,
    xAxisDimIndex,
    facetDimIndexes,
    {
      getXLabel: DimensionUtils.getXLabel,
      getBarValue: ChartDataUtils.getBarValue,
      getBarColor: DimensionUtils.getBarColor,
      getFacetKey: DimensionUtils.getFacetKey,
    },
  );
  return { facets, xAxisDimName, yAxisLabel };
}

function ChartVisual({ VisualClass, datumSet }) {
  const { facets, xAxisDimName, yAxisLabel } = useChartFacets(datumSet);

  return (
    <MultiChartLayout
      facets={facets}
      xAxisDimName={xAxisDimName}
      yAxisLabel={yAxisLabel}
      renderChart={({ data, xAxisLabel }) => (
        <VisualClass
          data={data}
          xAxisLabel={xAxisLabel}
          yAxisLabel={yAxisLabel}
        />
      )}
    />
  );
}

function VisualContent({ VisualClass, datumSet }) {
  if (VisualClass.IS_CHART) {
    return <ChartVisual VisualClass={VisualClass} datumSet={datumSet} />;
  }
  return <VisualClass datumSet={datumSet} />;
}

export default function VisualQueryPage() {
  const { "*": visualQueryStr } = useParams();
  const visualQuery = useMemo(
    () => VisualQuery.fromString(visualQueryStr),
    [visualQueryStr],
  );

  const [datumSet, setDatumSet] = useState(null);
  useEffect(() => {
    async function fetch() {
      setDatumSet(await Census2024.getDatumSetForQuery(visualQuery.query));
    }
    fetch();
  }, [visualQuery]);

  const VisualClass = visualQuery.visualClass;

  return (
    <Box sx={{ m: 1, p: 1 }}>
      <Typography variant="h6" sx={{ mt: 2, color: "info.main" }}>
        Lanka Data
      </Typography>
      <Typography variant="h4" sx={{ mt: 2 }}>
        {visualQueryStr}
      </Typography>
      {datumSet === null ? (
        <CircularProgress sx={{ m: 2 }} />
      ) : (
        <VisualContent VisualClass={VisualClass} datumSet={datumSet} />
      )}
    </Box>
  );
}
