import {
  buildSquareGrid,
  getBestSquareLabelFit,
  getSquareBoundaryEdges,
  getSquarePoints,
} from "../../../nonview/base/ShapeMapUtils.js";
import {
  buildShapeMapLayout,
  shareShapeMapScale,
} from "../../moles/visuals/ShapeMapLayoutUtils.js";
import ShapeMap from "./ShapeMap.js";

const SQUARE_SHAPE_CONFIG = {
  ariaLabel: "Square map",
  buildGrid: (bounds, totalCount) => {
    const { centers, size } = buildSquareGrid(bounds, totalCount);
    return { centers, shapeSize: size };
  },
  getBestLabelFit: getBestSquareLabelFit,
  getBoundaryEdges: getSquareBoundaryEdges,
  getExtent: (size) => size / 2,
  getPoints: getSquarePoints,
  name: "SquareMap",
  shapeName: "square",
  testId: "squaremap",
};

export function buildSquareMapLayout(
  facetInfo,
  valuePerSquare,
  isUnit = false,
) {
  const layout = buildShapeMapLayout(
    facetInfo,
    valuePerSquare,
    isUnit,
    SQUARE_SHAPE_CONFIG,
  );
  return { ...layout, squares: layout.shapes, size: layout.shapeSize };
}

export const shareSquareMapScale = shareShapeMapScale;

export default function SquareMap({ datumSet, isUnit = false }) {
  return (
    <ShapeMap
      datumSet={datumSet}
      isUnit={isUnit}
      shapeConfig={SQUARE_SHAPE_CONFIG}
    />
  );
}

SquareMap.IS_CHART = false;
