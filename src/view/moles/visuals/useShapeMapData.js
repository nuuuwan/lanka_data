import { useMemo } from "react";

import { getValuePerShape } from "../../../nonview/base/ShapeMapUtils.js";
import { SHAPE_MAP_MAX_SHAPES } from "../../_cons/MapCons.js";
import DimensionUtils from "../visual_utils/DimensionUtils.js";
import {
  buildFeatureToDataMap,
  groupDatumListByFacet,
} from "../visual_utils/GeoVisualUtils.js";
import {
  buildShapeMapFacetInfo,
  getLegendItems,
  getMatchedFeatures,
} from "./ShapeMapDataUtils.js";
import {
  buildShapeMapLayout,
  getShapeMapShapeCount,
  shareShapeMapScale,
} from "./ShapeMapLayoutUtils.js";

export default function useShapeMapData(
  geoJson,
  datumList,
  regionIndex,
  stackIndex,
  isUnit,
  shapeConfig,
) {
  return useMemo(() => {
    if (!geoJson) return { maps: [], legendItems: [] };
    const facetIndexes = DimensionUtils.getFacetDimIndexes(
      datumList,
      regionIndex,
      stackIndex,
    );
    const features = getMatchedFeatures(
      geoJson,
      datumList,
      regionIndex,
      stackIndex,
    );
    const groups = groupDatumListByFacet(datumList, facetIndexes);
    const facets = groups.map(({ facetKey, facetDatumList }) =>
      buildShapeMapFacetInfo(
        features,
        facetKey,
        buildFeatureToDataMap(facetDatumList, regionIndex, stackIndex),
      ),
    );
    const valuePerShape = isUnit
      ? null
      : getValuePerShape(
          facets.flatMap(({ regions }) => regions.map(({ weight }) => weight)),
          SHAPE_MAP_MAX_SHAPES,
        );
    const gridShapeCount = Math.max(
      ...facets.map((facet) =>
        getShapeMapShapeCount(facet, valuePerShape, isUnit),
      ),
    );
    const maps = DimensionUtils.sortFacets(
      shareShapeMapScale(
        facets.map((facet) =>
          buildShapeMapLayout(
            facet,
            valuePerShape,
            isUnit,
            shapeConfig,
            gridShapeCount,
          ),
        ),
      ),
      datumList,
      facetIndexes,
      (a, b) => b.total - a.total,
    );
    return { maps, legendItems: getLegendItems(facets) };
  }, [geoJson, datumList, regionIndex, stackIndex, isUnit, shapeConfig]);
}
