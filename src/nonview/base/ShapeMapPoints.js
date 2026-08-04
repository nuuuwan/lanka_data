export function getHexPoints([x, y], radius) {
  return Array.from({ length: 6 }, (_, pointIndex) => {
    const angle = Math.PI / 2 + (Math.PI / 3) * pointIndex;
    return [x + radius * Math.cos(angle), y + radius * Math.sin(angle)];
  });
}

export function getSquarePoints([x, y], size) {
  const halfSize = size / 2;
  return [
    [x - halfSize, y - halfSize],
    [x + halfSize, y - halfSize],
    [x + halfSize, y + halfSize],
    [x - halfSize, y + halfSize],
  ];
}
