export function iterCoordLists(geometry) {
  if (geometry.type === "Polygon") {
    return geometry.coordinates;
  }
  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.flat();
  }
  return [];
}

export function getFeatureId(feature) {
  return (
    feature.properties.region_id ??
    feature.properties.id ??
    feature.properties.name
  );
}

function getPolygonAreaAndCentroid(coords) {
  const n = coords.length;
  let area = 0;
  let cx = 0;
  let cy = 0;

  for (let i = 0; i < n; i++) {
    const [x0, y0] = coords[i];
    const [x1, y1] = coords[(i + 1) % n];
    const cross = x0 * y1 - x1 * y0;
    area += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }

  area /= 2;
  if (area === 0) {
    const xs = coords.map((c) => c[0]);
    const ys = coords.map((c) => c[1]);
    return {
      area: 0,
      centroid: [
        xs.reduce((a, b) => a + b, 0) / xs.length,
        ys.reduce((a, b) => a + b, 0) / ys.length,
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
  for (const poly of geometry.coordinates) {
    const { area, centroid } = getPolygonAreaAndCentroid(poly[0]);
    totalArea += area;
    totalCx += area * centroid[0];
    totalCy += area * centroid[1];
  }

  if (totalArea > 0) {
    return {
      area: totalArea,
      centroid: [totalCx / totalArea, totalCy / totalArea],
    };
  }
  return { area: 0, centroid: [0, 0] };
}

export function getGeometryStats(features) {
  const areas = {};
  const centroids = {};
  for (const feature of features) {
    const fid = getFeatureId(feature);
    const { area, centroid } = getFeatureAreaAndCentroid(feature.geometry);
    areas[fid] = area;
    centroids[fid] = centroid;
  }
  return { areas, centroids };
}
