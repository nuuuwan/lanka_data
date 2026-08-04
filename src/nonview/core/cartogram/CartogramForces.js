import { getFeatureId, iterCoordLists } from "./CartogramGeometry.js";

function getFij(dist, r, m) {
  if (dist > r) {
    return m * (r / dist);
  }
  const ratio = dist / r;
  return m * ((dist * dist) / (r * r)) * (4 - 3 * ratio);
}

export function computeForceParams(
  features,
  areas,
  weights,
  totalValue,
  totalArea,
) {
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

export function applyForces(features, centroids, radius, mass, frf) {
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
