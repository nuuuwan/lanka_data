function iterCoordLists(geometry) {
  if (geometry.type === "Polygon") {
    return geometry.coordinates;
  }
  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.flat();
  }
  return [];
}

function getFeatureId(feature) {
  return feature.properties.region_id ?? feature.properties.id;
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

function getGeometryStats(features) {
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

function loadWeights(features, regionIdToWeight) {
  const weights = {};
  for (const feature of features) {
    const fid = getFeatureId(feature);
    const w = regionIdToWeight[fid] ?? 1;
    if (w < 0) {
      throw new Error(`Negative PolygonValue for region ${fid}`);
    }
    weights[fid] = w;
  }

  const total = Object.values(weights).reduce((sum, w) => sum + w, 0);
  if (total === 0) {
    throw new Error("TotalValue is zero; all weights are zero.");
  }

  const normalized = {};
  for (const [fid, w] of Object.entries(weights)) {
    normalized[fid] = w / total;
  }
  return { weights: normalized, totalValue: 1 };
}

function getFij(dist, r, m) {
  if (dist > r) {
    return m * (r / dist);
  }
  const ratio = dist / r;
  return m * ((dist * dist) / (r * r)) * (4 - 3 * ratio);
}

function computeForceParams(features, areas, weights, totalValue, totalArea) {
  const radius = {};
  const mass = {};
  const sizeErrors = [];

  for (const feature of features) {
    const fid = getFeatureId(feature);
    const desired = totalArea * (weights[fid] / totalValue);
    const actual = areas[fid];
    const r = Math.sqrt(actual / Math.PI);
    const m = Math.sqrt(desired / Math.PI) - Math.sqrt(actual / Math.PI);
    const denom = Math.min(actual, desired);
    const sizeError = denom > 0 ? Math.max(actual, desired) / denom : 1;
    sizeErrors.push(sizeError);
    radius[fid] = r;
    mass[fid] = m;
  }

  const meanSizeError =
    sizeErrors.reduce((sum, e) => sum + e, 0) / sizeErrors.length;
  const frf = 1 / (1 + meanSizeError);
  return { radius, mass, frf, meanSizeError };
}

function displaceCoord(coord, polyCentroids, polyRadii, polyMasses, frf) {
  let dx = 0;
  let dy = 0;
  const [px, py] = coord;

  for (let i = 0; i < polyCentroids.length; i++) {
    const [cx, cy] = polyCentroids[i];
    const r = polyRadii[i];
    const m = polyMasses[i];
    const ddx = px - cx;
    const ddy = py - cy;
    const dist = Math.sqrt(ddx * ddx + ddy * ddy);
    if (dist === 0) {
      continue;
    }
    const angle = Math.atan2(ddy, ddx);
    const f = getFij(dist, r, m);
    dx += f * Math.cos(angle);
    dy += f * Math.sin(angle);
  }

  coord[0] += dx * frf;
  coord[1] += dy * frf;
}

function applyForces(features, centroids, radius, mass, frf) {
  const ids = features.map(getFeatureId);
  const polyCentroids = ids.map((fid) => centroids[fid]);
  const polyRadii = ids.map((fid) => radius[fid]);
  const polyMasses = ids.map((fid) => mass[fid]);

  for (const feature of features) {
    for (const coordList of iterCoordLists(feature.geometry)) {
      for (const coord of coordList) {
        displaceCoord(coord, polyCentroids, polyRadii, polyMasses, frf);
      }
    }
  }
}

export default class CartogramUtils {
  static EPSILON = 0.01;
  static MAX_ITERATIONS = 20;
  static MAX_TIME_MS = 10000;

  static compute(features, regionIdToWeight) {
    const { weights, totalValue } = loadWeights(features, regionIdToWeight);
    const startTime = performance.now();

    for (let i = 0; i < CartogramUtils.MAX_ITERATIONS; i++) {
      const { areas, centroids } = getGeometryStats(features);
      const totalArea = Object.values(areas).reduce((sum, a) => sum + a, 0);
      if (totalArea === 0) {
        break;
      }

      const { radius, mass, frf, meanSizeError } = computeForceParams(
        features,
        areas,
        weights,
        totalValue,
        totalArea,
      );

      const error = meanSizeError - 1;
      if (error < CartogramUtils.EPSILON) {
        break;
      }

      if (performance.now() - startTime > CartogramUtils.MAX_TIME_MS) {
        break;
      }

      applyForces(features, centroids, radius, mass, frf);
    }
  }
}
