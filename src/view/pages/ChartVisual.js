import ChartDataUtils from "../moles/visual_utils/ChartDataUtils.js";
import DimensionUtils from "../moles/visual_utils/DimensionUtils.js";
import MultiChartLayout from "../organisms/MultiChartLayout.js";

function getChartFacets(datumList, VisualClass) {
  if (!datumList.length)
    return {
      facets: [],
      xAxisDimName: "",
      yAxisLabel: "",
      stackDimIndex: null,
    };
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
  const facetIndexes = DimensionUtils.getFacetDimIndexes(
    datumList,
    xAxisDimIndex,
    stackDimIndex,
  );
  const helpers = {
    getXLabel: DimensionUtils.getXLabel,
    getBarValue: ChartDataUtils.getBarValue,
    getBarColor: DimensionUtils.getBarColor,
    getFacetKey: DimensionUtils.getFacetKey,
    getStackLabel: DimensionUtils.getStackLabel,
    getStackColor: DimensionUtils.getStackColor,
  };
  const facets =
    stackDimIndex === null
      ? ChartDataUtils.groupDataByFacet(
          datumList,
          xAxisDimIndex,
          facetIndexes,
          helpers,
        )
      : ChartDataUtils.groupStackedDataByFacet(
          datumList,
          xAxisDimIndex,
          stackDimIndex,
          facetIndexes,
          helpers,
        );
  return {
    facets,
    xAxisDimName: DimensionUtils.getDimName(datumList, xAxisDimIndex),
    yAxisLabel: datumList[0]?.query.aggregate ?? "",
    stackDimIndex,
  };
}

export default function ChartVisual({ VisualClass, datumSet }) {
  const info = getChartFacets(datumSet.datumList, VisualClass);
  const maxTotal = Math.max(...info.facets.map(({ total }) => total), 0);
  return (
    <MultiChartLayout
      facets={info.facets}
      xAxisDimName={info.xAxisDimName}
      yAxisLabel={info.yAxisLabel}
      fullWidth={VisualClass.IS_FULL_WIDTH}
      renderChart={({ data, total, xAxisLabel }) => (
        <VisualClass
          data={data}
          total={total}
          maxTotal={maxTotal}
          xAxisLabel={xAxisLabel}
          yAxisLabel={info.yAxisLabel}
          stackDimName={
            info.stackDimIndex !== null
              ? DimensionUtils.getDimName(
                  datumSet.datumList,
                  info.stackDimIndex,
                )
              : undefined
          }
        />
      )}
    />
  );
}
