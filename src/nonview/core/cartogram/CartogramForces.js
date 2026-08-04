import { getFeatureId } from "./CartogramGeometry.js";

function iterCoordLists(geometry) {
  if (geometry.type === "Polygon") {
    return geometry.coordinates;
  }
  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.flat();
  }
  return [];
}

function getForce(distance, radius, mass) {
  if (distance > radius) {
    return mass * (radius / distance);
  }
  const ratio = distance / radius;
  return mass * ((distance * distance) / (radius * radius)) * (4 - 3 * ratio);
}

function displaceCoord(coord, centroids, radii, masses, reductionFactor) {
  let dx = 0;
  let dy = 0;
  const [px, py] = coord;
  for (let index = 0; index < centroids.length; index++) {
    const [cx, cy] = centroids[index];
    const deltaX = px - cx;
    const deltaY = py - cy;
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    if (distance === 0) {
      continue;
    }
    const angle = Math.atan2(deltaY, deltaX);
    const force = getForce(distance, radii[index], masses[index]);
    dx += force * Math.cos(angle);
    dy += force * Math.sin(angle);
  }
  coord[0] += dx * reductionFactor;
  coord[1] += dy * reductionFactor;
}

export function applyForces(features, centroids, radius, mass, factor) {
  const featureIds = features.map(getFeatureId);
  const polygonCentroids = featureIds.map((id) => centroids[id]);
  const polygonRadii = featureIds.map((id) => radius[id]);
  const polygonMasses = featureIds.map((id) => mass[id]);
  for (const feature of features) {
    for (const coordList of iterCoordLists(feature.geometry)) {
      for (const coord of coordList) {
        displaceCoord(
          coord,
          polygonCentroids,
          polygonRadii,
          polygonMasses,
          factor,
        );
      }
    }
  }
}
