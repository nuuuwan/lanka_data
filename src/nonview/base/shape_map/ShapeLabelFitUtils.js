function getAxes(angleDegrees) {
  const angle = (angleDegrees * Math.PI) / 180;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [
    [cosine, sine],
    [-sine, cosine],
  ];
}

function projectPoint(point, axis) {
  return point[0] * axis[0] + point[1] * axis[1];
}

function getLongestRun(items, step) {
  const sortedItems = [...items].sort((left, right) => left[0] - right[0]);
  let run = [sortedItems[0]];
  let longestRun = run;
  const tolerance = step * 0.5;
  for (let index = 1; index < sortedItems.length; index += 1) {
    const previous = sortedItems[index - 1];
    const current = sortedItems[index];
    run =
      Math.abs(current[0] - previous[0] - step) < tolerance
        ? [...run, current]
        : [current];
    if (run.length > longestRun.length) {
      longestRun = run;
    }
  }
  return longestRun;
}

function getBestLabelFit(points, size, angles, lineSpacing) {
  let best = null;
  for (const angle of angles) {
    const [horizontalAxis, verticalAxis] = getAxes(angle);
    const lines = new Map();
    for (const point of points) {
      const key = Math.round(projectPoint(point, verticalAxis) / lineSpacing);
      const line = lines.get(key) ?? [];
      line.push([projectPoint(point, horizontalAxis), point]);
      lines.set(key, line);
    }
    for (const line of lines.values()) {
      const run = getLongestRun(line, size);
      if (!best || run.length > best.run.length) {
        best = { angle, run };
      }
    }
  }
  const first = best.run[0][1];
  const last = best.run.at(-1)[1];
  return {
    center: [(first[0] + last[0]) / 2, (first[1] + last[1]) / 2],
    width: best.run.length * size,
    height: lineSpacing,
    angle: best.angle,
  };
}

export function getBestHexLabelFit(points, radius) {
  return getBestLabelFit(
    points,
    Math.sqrt(3) * radius,
    [0, 60, -60],
    1.5 * radius,
  );
}

export function getBestSquareLabelFit(points, size) {
  return getBestLabelFit(points, size, [0, 90], size);
}
