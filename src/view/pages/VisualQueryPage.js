import { useNavigate, useParams } from "react-router-dom";
import { Typography, Box, LinearProgress, TextField } from "@mui/material";
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
  useEffect(() => {
    console.debug(
      `[VisualQueryPage] Displaying ${VisualClass.name} with ${datumSet.datumList.length} datums`,
    );
  }, [VisualClass, datumSet]);

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
  const navigate = useNavigate();
  const { isReady } = useContext(DataContext);
  const [visualQueryInput, setVisualQueryInput] = useState(visualQueryStr);

  useEffect(() => {
    setVisualQueryInput(visualQueryStr);
  }, [visualQueryStr]);

  function submitVisualQuery(event) {
    event.preventDefault();
    const nextVisualQueryStr = visualQueryInput.trim();
    if (nextVisualQueryStr && nextVisualQueryStr !== visualQueryStr) {
      navigate(`/${nextVisualQueryStr}`);
    }
  }

  const [visualQuery, setVisualQuery] = useState(null);
  useEffect(() => {
    if (!isReady) {
      console.debug(
        `[VisualQueryPage] Waiting for application data before parsing "${visualQueryStr}"`,
      );
      return;
    }
    async function parse() {
      console.debug(`[VisualQueryPage] Parsing "${visualQueryStr}"`);
      const nextVisualQuery = await VisualQuery.fromString(visualQueryStr);
      console.debug(
        `[VisualQueryPage] Parsed "${visualQueryStr}" as ${nextVisualQuery.visualClass.name}`,
      );
      setVisualQuery(nextVisualQuery);
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
      console.debug(
        `[VisualQueryPage] Fetching data for "${visualQuery.query}"`,
      );
      const startTime = performance.now();
      const nextDatumSet = await DataSourceFactory.getDatumSetForQuery(
        visualQuery.query,
      );
      const nextLoadTimeSeconds = (performance.now() - startTime) / 1000;
      if (!cancelled) {
        setDatumSet(nextDatumSet);
        setLoadTimeSeconds(nextLoadTimeSeconds);
        console.debug(
          `[VisualQueryPage] Data ready: ${nextDatumSet.datumList.length} datums in ${nextLoadTimeSeconds.toFixed(3)}s`,
        );
      } else {
        console.debug(
          `[VisualQueryPage] Ignoring completed data fetch for stale query "${visualQuery.query}"`,
        );
      }
    }
    fetch();
    return () => {
      cancelled = true;
      console.debug(
        `[VisualQueryPage] Cancelling data update for "${visualQuery.query}"`,
      );
    };
  }, [visualQuery]);

  const VisualClass = visualQuery?.visualClass;

  return (
    <Box sx={{ m: 2 }}>
      <Box component="form" onSubmit={submitVisualQuery} sx={{ mb: 2 }}>
        <TextField
          fullWidth
          label="Visual query"
          size="small"
          value={visualQueryInput}
          onChange={(event) => setVisualQueryInput(event.target.value)}
          helperText="Press Enter to update"
          slotProps={{
            htmlInput: {
              autoComplete: "off",
              spellCheck: false,
            },
          }}
          sx={{
            "& .MuiInputBase-input": {
              fontFamily: "monospace",
            },
          }}
        />
      </Box>
      {!isReady || datumSet === null || loadTimeSeconds === null ? (
        <LinearProgress sx={{ m: 2 }} />
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
