import { geoCentroid } from "d3-geo";

import {
  assignShapes,
  getShapeCounts,
} from "../../../nonview/base/ShapeMapUtils.js";
import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import { MAP_HEIGHT, MAP_PADDING, MAP_WIDTH } from "../../_cons/MapCons.js";
import { getProjectionInfo } from "../visual_utils/GeoVisualUtils.js";
import { getShapeMapLabels, getShapeMapViewBox } from "./ShapeMapLabelUtils.js";

function getCounts(facetInfo, valuePerShape, isUnit) {
  if (isUnit) {
    return Object.fromEntries(facetInfo.regions.map(({ id }) => [id, 1]));
  }
  const weights = Object.fromEntries(
    facetInfo.regions.map(({ id, weight }) => [id, weight]),
  );
  return getShapeCounts(weights, valuePerShape);
}

export function getShapeMapShapeCount(facetInfo, valuePerShape, isUnit) {
  return Object.values(getCounts(facetInfo, valuePerShape, isUnit)).reduce(
    (sum, count) => sum + count,
    0,
  );
}

export function buildShapeMapLayout(
  facetInfo,
  valuePerShape,
  isUnit,
  shapeConfig,
  gridShapeCount = null,
) {
  const counts = getCounts(facetInfo, valuePerShape, isUnit);
  const visibleRegions = facetInfo.regions.filter(({ id }) => counts[id] > 0);
  const features = visibleRegions.map(({ feature }) =>
    JSON.parse(JSON.stringify(feature)),
  );
  const visibleWeights = Object.fromEntries(
    visibleRegions.map(({ id, weight }) => [id, weight]),
  );
  if (!isUnit && Object.values(visibleWeights).some((weight) => weight > 0))
    CartogramUtils.compute(features, visibleWeights);
  const { projection } = getProjectionInfo(features);
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
    gridShapeCount ?? totalCount,
  );
  const regions = features.map((feature, index) => {
    const id = visibleRegions[index].id;
    return {
      centroid: projection(geoCentroid(feature)),
      count: counts[id],
      id,
    };
  });
  const assigned = assignShapes(regions, centers);
  const regionById = new Map(
    visibleRegions.map((region) => [region.id, region]),
  );
  const shapes = assigned.map(({ id, center }, index) => ({
    ...regionById.get(id),
    center,
    id: `${id}-${index}`,
    points: shapeConfig.getPoints(center, shapeSize),
    regionId: id,
  }));
  const facetColor = visibleRegions
    .map(({ display }) => display.color)
    .find(
      (color, index, colors) => color && colors.every((item) => item === color),
    );
  const shapeValues = visibleRegions.map(
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
    omittedBelow:
      visibleRegions.length < facetInfo.regions.length
        ? valuePerShape / 2
        : null,
    shapeSize,
    shapes,
    facetColor,
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
  const minX = Math.min(...maps.map(({ viewBox }) => viewBox[0]));
  const minY = Math.min(...maps.map(({ viewBox }) => viewBox[1]));
  const maxX = Math.max(...maps.map(({ viewBox }) => viewBox[0] + viewBox[2]));
  const maxY = Math.max(...maps.map(({ viewBox }) => viewBox[1] + viewBox[3]));
  const viewBox = [minX, minY, maxX - minX, maxY - minY];
  return maps.map((map) => ({
    ...map,
    shapeValueMin,
    shapeValueMax,
    viewBox,
  }));
}
