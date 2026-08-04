function getSeriesKeys(data) {
  if (data.length === 0) {
    return [];
  }
  return Object.keys(data[0]).filter(
    (key) => key !== "id" && key !== "_barWidth" && !key.endsWith("Color"),
  );
}

function getMixingScore(left, right, seriesKeys) {
  let score = 0;
  for (let i = 0; i < seriesKeys.length; i++) {
    for (let j = i + 1; j < seriesKeys.length; j++) {
      const firstKey = seriesKeys[i];
      const secondKey = seriesKeys[j];
      const leftOrder = Math.sign(
        (left[firstKey] ?? 0) - (left[secondKey] ?? 0),
      );
      const rightOrder = Math.sign(
        (right[firstKey] ?? 0) - (right[secondKey] ?? 0),
      );
      if (leftOrder !== 0 && rightOrder !== 0 && leftOrder !== rightOrder) {
        score++;
      }
    }
  }
  return score;
}

function getPathMixingScore(path, seriesKeys) {
  return path
    .slice(1)
    .reduce(
      (score, row, index) =>
        score + getMixingScore(path[index], row, seriesKeys),
      0,
    );
}

export function sortAreaBumpXAxis(data) {
  if (data.length < 2) {
    return [...data];
  }
  const seriesKeys = getSeriesKeys(data);
  let bestPath = [...data];
  let bestScore = getPathMixingScore(bestPath, seriesKeys);
  for (let startIndex = 0; startIndex < data.length; startIndex++) {
    const path = [data[startIndex]];
    const remaining = data
      .map((row, index) => ({ row, index }))
      .filter(({ index }) => index !== startIndex);
    while (remaining.length > 0) {
      const previous = path.at(-1);
      let nearestIndex = 0;
      let nearestScore = getMixingScore(previous, remaining[0].row, seriesKeys);
      for (let i = 1; i < remaining.length; i++) {
        const score = getMixingScore(previous, remaining[i].row, seriesKeys);
        if (score < nearestScore) {
          nearestIndex = i;
          nearestScore = score;
        }
      }
      path.push(remaining.splice(nearestIndex, 1)[0].row);
    }
    const score = getPathMixingScore(path, seriesKeys);
    if (score < bestScore) {
      bestPath = path;
      bestScore = score;
    }
  }
  return bestPath;
}

export function toAreaBumpData(data) {
  const sortedData = sortAreaBumpXAxis(data);
  return getSeriesKeys(sortedData).map((key) => ({
    id: key,
    color: sortedData.find((row) => row[`${key}Color`])?.[`${key}Color`],
    data: sortedData.map((row) => ({ x: row.id, y: row[key] ?? 0 })),
  }));
}
