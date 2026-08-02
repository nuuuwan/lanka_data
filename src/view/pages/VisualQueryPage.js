import { useParams } from "react-router-dom";
import { Typography, Box, CircularProgress } from "@mui/material";
import { useState, useEffect } from "react";
import DataSourceFactory from "../../nonview/core/data_source/DataSourceFactory.js";
import VisualQuery from "../../nonview/core/VisualQuery.js";
import Region from "../../nonview/core/thing/concept/category_concept/region/region/Region.js";
import ChartDataUtils from "../moles/visual_utils/ChartDataUtils.js";
import DimensionUtils from "../moles/visual_utils/DimensionUtils.js";
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

function VisualContent({ VisualClass, datumSet }) {
  if (VisualClass.IS_CHART) {
    return <ChartVisual VisualClass={VisualClass} datumSet={datumSet} />;
  }
  return <VisualClass datumSet={datumSet} />;
}

export default function VisualQueryPage() {
  const { "*": visualQueryStr } = useParams();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    Region.init().then(() => setIsReady(true));
  }, []);

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
  useEffect(() => {
    if (!visualQuery) {
      return;
    }
    async function fetch() {
      setDatumSet(
        await DataSourceFactory.getDatumSetForQuery(visualQuery.query),
      );
    }
    fetch();
  }, [visualQuery]);

  const VisualClass = visualQuery?.visualClass;

  return (
    <Box sx={{ m: 2 }}>
      <Typography variant="h6" sx={{ color: "info.main" }}>
        Lanka Data
      </Typography>
      <Typography variant="h6" sx={{ color: "primary" }}>
        {visualQueryStr}
      </Typography>
      {!isReady || datumSet === null ? (
        <CircularProgress sx={{ m: 2 }} />
      ) : (
        <Box data-testid="visual-content">
          <VisualContent VisualClass={VisualClass} datumSet={datumSet} />
        </Box>
      )}
    </Box>
  );
}
