import { useMemo } from "react";

import { getValuePerShape } from "../../../nonview/base/ShapeMapUtils.js";
import { HEX_MAP_MAX_HEXAGONS } from "../../../nonview/constants/MAP.js";
import DimensionUtils from "../../../nonview/core/visual/DimensionUtils.js";
import {
  buildFeatureToDataMap,
  groupDatumListByFacet,
} from "../../../nonview/core/visual/GeoVisualUtils.js";
import {
  buildShapeMapFacetInfo,
  getLegendItems,
  getMatchedFeatures,
} from "./ShapeMapDataUtils.js";
import {
  buildShapeMapLayout,
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
          HEX_MAP_MAX_HEXAGONS,
        );
    const maps = DimensionUtils.sortFacets(
      shareShapeMapScale(
        facets.map((facet) =>
          buildShapeMapLayout(facet, valuePerShape, isUnit, shapeConfig),
        ),
      ),
      datumList,
      facetIndexes,
      (a, b) => b.total - a.total,
    );
    return { maps, legendItems: getLegendItems(facets) };
  }, [geoJson, datumList, regionIndex, stackIndex, isUnit, shapeConfig]);
}
