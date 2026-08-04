import { geoCentroid } from "d3-geo";

import {
  assignShapes,
  getShapeCounts,
} from "../../../nonview/base/ShapeMapUtils.js";
import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import {
  MAP_HEIGHT,
  MAP_PADDING,
  MAP_WIDTH,
} from "../../../nonview/constants/MAP.js";
import { getProjectionInfo } from "../../../nonview/core/visual/GeoVisualUtils.js";
import { getShapeMapLabels, getShapeMapViewBox } from "./ShapeMapLabelUtils.js";

export function buildShapeMapLayout(
  facetInfo,
  valuePerShape,
  isUnit,
  shapeConfig,
) {
  const features = facetInfo.regions.map(({ feature }) =>
    JSON.parse(JSON.stringify(feature)),
  );
  const weights = Object.fromEntries(
    facetInfo.regions.map(({ id, weight }) => [id, weight]),
  );
  if (!isUnit && Object.values(weights).some((weight) => weight > 0))
    CartogramUtils.compute(features, weights);
  const { projection } = getProjectionInfo(features);
  const counts = isUnit
    ? Object.fromEntries(facetInfo.regions.map(({ id }) => [id, 1]))
    : getShapeCounts(weights, valuePerShape);
  const totalCount = Object.values(counts).reduce(
    (sum, count) => sum + count,
    0,
  );
  const { centers, shapeSize } = shapeConfig.buildGrid(
    [
      MAP_PADDING,
      MAP_PADDING,
      MAP_WIDTH - MAP_PADDING,
      MAP_HEIGHT - MAP_PADDING,
    ],
    totalCount,
  );
  const regions = features.map((feature, index) => {
    const id = facetInfo.regions[index].id;
    return {
      centroid: projection(geoCentroid(feature)),
      count: counts[id],
      id,
    };
  });
  const assigned = assignShapes(regions, centers);
  const regionById = new Map(
    facetInfo.regions.map((region) => [region.id, region]),
  );
  const shapes = assigned.map(({ id, center }, index) => ({
    ...regionById.get(id),
    center,
    id: `${id}-${index}`,
    points: shapeConfig.getPoints(center, shapeSize),
    regionId: id,
  }));
  const shapeValues = facetInfo.regions.map(
    ({ id, weight }) => weight / counts[id],
  );
  return {
    boundaryEdges: shapeConfig.getBoundaryEdges(assigned, shapeSize),
    facetKey: facetInfo.facetKey,
    labels: getShapeMapLabels(
      assigned,
      regionById,
      shapeSize,
      shapeConfig.getBestLabelFit,
    ),
    shapeSize,
    shapes,
    shapeValueMax: Math.max(...shapeValues),
    shapeValueMin: Math.min(...shapeValues),
    total: facetInfo.regions.reduce((sum, { weight }) => sum + weight, 0),
    viewBox: getShapeMapViewBox(shapes, shapeConfig.getExtent(shapeSize)),
  };
}

export function shareShapeMapScale(maps) {
  if (!maps.length) return maps;
  const shapeValueMin = Math.min(...maps.map((map) => map.shapeValueMin));
  const shapeValueMax = Math.max(...maps.map((map) => map.shapeValueMax));
  return maps.map((map) => ({ ...map, shapeValueMin, shapeValueMax }));
}
