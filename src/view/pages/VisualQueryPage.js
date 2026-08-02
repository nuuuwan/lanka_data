import { useParams } from "react-router-dom";
import { Typography, Box, CircularProgress } from "@mui/material";
import { useState, useEffect, useContext } from "react";
import DataSourceFactory from "../../nonview/core/data_source/DataSourceFactory.js";
import VisualQuery from "../../nonview/core/VisualQuery.js";
import DataContext from "../../nonview/core/data_context/DataContext.js";
import ChartDataUtils from "../moles/visual_utils/ChartDataUtils.js";
import DimensionUtils from "../moles/visual_utils/DimensionUtils.js";
import FormatUtils from "../moles/visual_utils/FormatUtils.js";
import MultiChartLayout from "../moles/visual_utils/MultiChartLayout.js";
function useChartFacets(datumSet, VisualClass) {
  const { datumList } = datumSet;

  if (datumList.length === 0) {
    return {
      facets: [],
      xAxisDimName: "",
      yAxisLabel: "",
      stackDimIndex: null,
    };
  }

  let xAxisDimIndex;
  let stackDimIndex;
  if (VisualClass.IS_MARIMEKKO) {
    ({ xAxisDimIndex, stackDimIndex } =
      DimensionUtils.getMarimekkoDimIndexes(datumList));
  } else {
    stackDimIndex = VisualClass.IS_STACKED
      ? DimensionUtils.getStackDimIndex(datumList)
      : null;
    xAxisDimIndex = DimensionUtils.getXAxisDimIndex(datumList, stackDimIndex);
  }

  const facetDimIndexes = DimensionUtils.getFacetDimIndexes(
    datumList,
    xAxisDimIndex,
    stackDimIndex,
  );
  const xAxisDimName = DimensionUtils.getDimName(datumList, xAxisDimIndex);
  const yAxisLabel = datumList[0]?.query.aggregate ?? "";

  if (stackDimIndex === null) {
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
    return { facets, xAxisDimName, yAxisLabel, stackDimIndex };
  }

  const facets = ChartDataUtils.groupStackedDataByFacet(
    datumList,
    xAxisDimIndex,
    stackDimIndex,
    facetDimIndexes,
    {
      getXLabel: DimensionUtils.getXLabel,
      getStackLabel: DimensionUtils.getStackLabel,
      getStackColor: DimensionUtils.getStackColor,
      getBarValue: ChartDataUtils.getBarValue,
      getFacetKey: DimensionUtils.getFacetKey,
    },
  );
  return { facets, xAxisDimName, yAxisLabel, stackDimIndex };
}

function ChartVisual({ VisualClass, datumSet }) {
  const { facets, xAxisDimName, yAxisLabel, stackDimIndex } = useChartFacets(
    datumSet,
    VisualClass,
  );

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
          stackDimName={
            stackDimIndex !== null
              ? DimensionUtils.getDimName(datumSet.datumList, stackDimIndex)
              : undefined
          }
        />
      )}
    />
  );
}

function VisualContent({ VisualClass, datumSet, loadTimeSeconds }) {
  return (
    <>
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
    </>
  );
}

export default function VisualQueryPage() {
  const { "*": visualQueryStr } = useParams();
  const { isReady } = useContext(DataContext);

  const [visualQuery, setVisualQuery] = useState(null);
  useEffect(() => {
    if (!isReady) {
      return;
    }
    async function parse() {
      setVisualQuery(await VisualQuery.fromString(visualQueryStr));
    }
    parse();
  }, [isReady, visualQueryStr]);

  const [datumSet, setDatumSet] = useState(null);
  const [loadTimeSeconds, setLoadTimeSeconds] = useState(null);
  useEffect(() => {
    if (!visualQuery) {
      return;
    }
    let cancelled = false;
    async function fetch() {
      setDatumSet(null);
      setLoadTimeSeconds(null);
      const startTime = performance.now();
      const nextDatumSet = await DataSourceFactory.getDatumSetForQuery(
        visualQuery.query,
      );
      if (!cancelled) {
        setDatumSet(nextDatumSet);
        setLoadTimeSeconds((performance.now() - startTime) / 1000);
      }
    }
    fetch();
    return () => {
      cancelled = true;
    };
  }, [visualQuery]);

  const VisualClass = visualQuery?.visualClass;

  return (
    <Box sx={{ m: 2 }}>
      <Typography
        variant="h6"
        sx={{ color: "primary", overflowWrap: "anywhere" }}
      >
        {visualQueryStr}
      </Typography>
      {!isReady || datumSet === null || loadTimeSeconds === null ? (
        <CircularProgress sx={{ m: 2 }} />
      ) : (
        <Box data-testid="visual-content">
          <VisualContent
            VisualClass={VisualClass}
            datumSet={datumSet}
            loadTimeSeconds={loadTimeSeconds}
          />
        </Box>
      )}
    </Box>
  );
}
