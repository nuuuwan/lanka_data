import { useNavigate, useParams } from "react-router-dom";
import { Alert, AlertTitle, Typography, Box } from "@mui/material";
import { useState, useEffect, useContext, useRef } from "react";
import DataSourceFactory from "../../nonview/core/data_source/DataSourceFactory.js";
import VisualQuery from "../../nonview/core/VisualQuery.js";
import DataContext from "../../nonview/core/data_context/DataContext.js";
import ChartDataUtils from "../moles/visual_utils/ChartDataUtils.js";
import DimensionUtils from "../moles/visual_utils/DimensionUtils.js";
import FormatUtils from "../moles/visual_utils/FormatUtils.js";
import LoadingProgress from "../molecules/LoadingProgress.js";
import DataProvenancePanel from "../molecules/DataProvenancePanel.js";
import MultiChartLayout from "../organisms/MultiChartLayout.js";
import VisualErrorBoundary from "../organisms/VisualErrorBoundary.js";
import ChangeViewSection from "../organisms/ChangeViewSection.js";
import VisualQueryForm from "../organisms/VisualQueryForm.js";
import RecentQueriesMenu from "../organisms/RecentQueriesMenu.js";
import VisualHeading from "../molecules/VisualHeading.js";
import styles from "./VisualQueryPage.module.css";

function getElapsedTimeSeconds(startTime, currentTime) {
  return startTime === null ? 0 : Math.max(0, currentTime - startTime) / 1000;
}

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

function VisualContent({ VisualClass, datumSet, loadTimeSeconds, query }) {
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

export default function VisualQueryPage() {
  const { "*": visualQueryStr } = useParams();
  const navigate = useNavigate();
  const { isReady, queryOptions } = useContext(DataContext);
  const [visualQueryInput, setVisualQueryInput] = useState(visualQueryStr);
  const [errorMessage, setErrorMessage] = useState(null);
  const applicationLoadStartTime = useRef(performance.now());
  const [applicationLoadTimeSeconds, setApplicationLoadTimeSeconds] = useState(
    isReady ? 0 : null,
  );
  const [currentTime, setCurrentTime] = useState(() => performance.now());
  const parseStartTime = useRef(null);
  const dataLoadStartTime = useRef(null);
  const visualRef = useRef(null);

  useEffect(() => {
    if (isReady && applicationLoadTimeSeconds === null) {
      setApplicationLoadTimeSeconds(
        (performance.now() - applicationLoadStartTime.current) / 1000,
      );
    }
  }, [applicationLoadTimeSeconds, isReady]);

  useEffect(() => {
    setVisualQueryInput(visualQueryStr);
  }, [visualQueryStr]);

  function submitVisualQuery() {
    const nextVisualQueryStr = visualQueryInput.trim();
    if (nextVisualQueryStr && nextVisualQueryStr !== visualQueryStr) {
      setDatumSet(null);
      setLoadTimeSeconds(null);
      navigate(`/${nextVisualQueryStr}`);
    }
  }

  const [visualQuery, setVisualQuery] = useState(null);
  const [parseTimeSeconds, setParseTimeSeconds] = useState(null);
  useEffect(() => {
    if (!isReady) {
      console.debug(
        `[VisualQueryPage] Waiting for application data before parsing "${visualQueryStr}"`,
      );
      return;
    }
    let cancelled = false;
    async function parse() {
      console.debug(`[VisualQueryPage] Parsing "${visualQueryStr}"`);
      setVisualQuery(null);
      setParseTimeSeconds(null);
      setDatumSet(null);
      setLoadTimeSeconds(null);
      setErrorMessage(null);
      const startTime = performance.now();
      parseStartTime.current = startTime;
      setCurrentTime(startTime);
      try {
        const nextVisualQuery = await VisualQuery.fromString(visualQueryStr);
        if (!cancelled) {
          setParseTimeSeconds((performance.now() - startTime) / 1000);
          console.debug(
            `[VisualQueryPage] Parsed "${visualQueryStr}" as ${nextVisualQuery.visualClass.name}`,
          );
          setVisualQuery(nextVisualQuery);
        }
      } catch (error) {
        console.error("[VisualQueryPage] Could not parse request", error);
        if (!cancelled) {
          setErrorMessage(
            "We couldn't understand that request. Please check your choices and try again.",
          );
        }
      }
    }
    parse();
    return () => {
      cancelled = true;
    };
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
      dataLoadStartTime.current = startTime;
      setCurrentTime(startTime);
      try {
        const nextDatumSet = await DataSourceFactory.getDatumSetForQuery(
          visualQuery.query,
        );
        const nextLoadTimeSeconds = (performance.now() - startTime) / 1000;
        if (!cancelled) {
          setDatumSet(nextDatumSet);
          setLoadTimeSeconds(nextLoadTimeSeconds);
          setErrorMessage(
            nextDatumSet.datumList.length === 0
              ? "We couldn't find any data for that request. Please check your choices and try again."
              : null,
          );
          console.debug(
            `[VisualQueryPage] Data ready: ${nextDatumSet.datumList.length} datums in ${nextLoadTimeSeconds.toFixed(3)}s`,
          );
        } else {
          console.debug(
            `[VisualQueryPage] Ignoring completed data fetch for stale query "${visualQuery.query}"`,
          );
        }
      } catch (error) {
        console.error("[VisualQueryPage] Could not load requested data", error);
        if (!cancelled) {
          setErrorMessage(
            "We couldn't load the data for that request. Please try again.",
          );
        }
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
  const isLoading =
    !isReady || !VisualClass || datumSet === null || loadTimeSeconds === null;

  useEffect(() => {
    if (!isLoading && !errorMessage) {
      visualRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [errorMessage, isLoading]);

  useEffect(() => {
    if (!isLoading) {
      return;
    }
    let animationFrameId;
    function updateCurrentTime() {
      setCurrentTime(performance.now());
      animationFrameId = requestAnimationFrame(updateCurrentTime);
    }
    animationFrameId = requestAnimationFrame(updateCurrentTime);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isLoading]);

  const loadingSteps = [
    {
      label: "Loading application data",
      status: isReady ? "complete" : "active",
      durationSeconds:
        applicationLoadTimeSeconds ??
        getElapsedTimeSeconds(applicationLoadStartTime.current, currentTime),
    },
    {
      label: "Understanding request",
      status: !isReady ? "pending" : VisualClass ? "complete" : "active",
      durationSeconds:
        parseTimeSeconds ??
        getElapsedTimeSeconds(parseStartTime.current, currentTime),
    },
    {
      label: "Loading visual data",
      status: !VisualClass
        ? "pending"
        : datumSet === null || loadTimeSeconds === null
          ? "active"
          : "complete",
      durationSeconds:
        loadTimeSeconds ??
        getElapsedTimeSeconds(dataLoadStartTime.current, currentTime),
    },
  ];

  return (
    <Box className={styles.page}>
      <VisualQueryForm
        value={visualQueryInput}
        onChange={setVisualQueryInput}
        onSubmit={submitVisualQuery}
        queryOptions={queryOptions}
      />
      <ExampleQueryGallery />
      <RecentQueriesMenu
        loadedVisualQuery={
          datumSet?.datumList.length > 0 && loadTimeSeconds !== null
            ? visualQueryStr
            : null
        }
      />
      {errorMessage ? (
        <Alert severity="error" data-testid="query-error">
          <AlertTitle>Sorry, something went wrong.</AlertTitle>
          {errorMessage}
        </Alert>
      ) : (
        <Box
          className={styles.visual}
          data-testid={isLoading ? undefined : "visual-content"}
          ref={visualRef}
        >
          {isLoading ? (
            <LoadingProgress steps={loadingSteps} />
          ) : (
            <VisualErrorBoundary key={visualQueryStr}>
              <VisualContent
                VisualClass={VisualClass}
                datumSet={datumSet}
                loadTimeSeconds={loadTimeSeconds}
                query={visualQuery.query}
              />
            </VisualErrorBoundary>
          )}
        </Box>
      )}
      {(errorMessage || !isLoading) && (
        <ChangeViewSection
          value={visualQueryInput}
          onChange={setVisualQueryInput}
          onSubmit={submitVisualQuery}
          queryOptions={queryOptions}
          loadedVisualQuery={
            datumSet?.datumList.length > 0 && loadTimeSeconds !== null
              ? visualQueryStr
              : null
          }
        />
      )}
    </Box>
  );
}
