function getPolygonAreaAndCentroid(coords) {
  const pointCount = coords.length;
  let area = 0;
  let cx = 0;
  let cy = 0;
  for (let index = 0; index < pointCount; index++) {
    const [x0, y0] = coords[index];
    const [x1, y1] = coords[(index + 1) % pointCount];
    const cross = x0 * y1 - x1 * y0;
    area += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  area /= 2;
  if (area === 0) {
    const xs = coords.map(([x]) => x);
    const ys = coords.map(([, y]) => y);
    return {
      area: 0,
      centroid: [
        xs.reduce((sum, value) => sum + value, 0) / xs.length,
        ys.reduce((sum, value) => sum + value, 0) / ys.length,
      ],
    };
  }
  return {
    area: Math.abs(area),
    centroid: [cx / (6 * area), cy / (6 * area)],
  };
}

function getFeatureAreaAndCentroid(geometry) {
  if (geometry.type === "Polygon") {
    return getPolygonAreaAndCentroid(geometry.coordinates[0]);
  }
  let totalArea = 0;
  let totalCx = 0;
  let totalCy = 0;
  for (const polygon of geometry.coordinates) {
    const { area, centroid } = getPolygonAreaAndCentroid(polygon[0]);
    totalArea += area;
    totalCx += area * centroid[0];
    totalCy += area * centroid[1];
  }
  return totalArea > 0
    ? {
        area: totalArea,
        centroid: [totalCx / totalArea, totalCy / totalArea],
      }
    : { area: 0, centroid: [0, 0] };
}

export function getFeatureId(feature) {
  return (
    feature.properties.region_id ??
    feature.properties.id ??
    feature.properties.name
  );
}

export function getGeometryStats(features) {
  const areas = {};
  const centroids = {};
  for (const feature of features) {
    const featureId = getFeatureId(feature);
    const { area, centroid } = getFeatureAreaAndCentroid(feature.geometry);
    areas[featureId] = area;
    centroids[featureId] = centroid;
  }
  return { areas, centroids };
}
