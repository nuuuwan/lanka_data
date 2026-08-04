import {
  buildHexGrid,
  getBestHexLabelFit,
  getHexBoundaryEdges,
  getHexPoints,
} from "../../../nonview/base/ShapeMapUtils.js";
import {
  buildShapeMapLayout,
  shareShapeMapScale,
} from "../../moles/visuals/ShapeMapLayoutUtils.js";
import ShapeMap from "./ShapeMap.js";

const HEX_SHAPE_CONFIG = {
  ariaLabel: "Hex map",
  buildGrid: (bounds, totalCount) => {
    const { centers, radius } = buildHexGrid(bounds, totalCount);
    return { centers, shapeSize: radius };
  },
  getBestLabelFit: getBestHexLabelFit,
  getBoundaryEdges: getHexBoundaryEdges,
  getExtent: (radius) => radius,
  getPoints: getHexPoints,
  name: "HexMap",
  shapeName: "hexagon",
  testId: "hexmap",
};

export function buildHexMapLayout(facetInfo, valuePerHexagon, isUnit = false) {
  const layout = buildShapeMapLayout(
    facetInfo,
    valuePerHexagon,
    isUnit,
    HEX_SHAPE_CONFIG,
  );
  return { ...layout, hexagons: layout.shapes, radius: layout.shapeSize };
}

export const shareHexMapScale = shareShapeMapScale;

export default function HexMap({ datumSet, isUnit = false }) {
  return (
    <ShapeMap
      datumSet={datumSet}
      isUnit={isUnit}
      shapeConfig={HEX_SHAPE_CONFIG}
    />
  );
}

HexMap.IS_CHART = false;
