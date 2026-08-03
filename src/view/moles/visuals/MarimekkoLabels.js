const getFontScale = (screenWidth) => screenWidth / 1200;

export function BarLabelsLayer({ data, screenWidth }) {
  const congestionScale = Math.max(0.7, 1 / Math.sqrt(data.length));
  const fontScale = getFontScale(screenWidth) * congestionScale;
  return (
    <>
      {data.map((datum) =>
        datum.width < 20 ? null : (
          <text
            key={datum.id}
            x={datum.x + datum.width / 2}
            y={datum.y + datum.height + 12}
            textAnchor="middle"
            dominantBaseline="hanging"
            style={{
              fontSize: Math.max(
                6,
                Math.min(10, (datum.width / 10) * fontScale),
              ),
              fill: "#333",
            }}
          >
            {datum.id}
          </text>
        ),
      )}
    </>
  );
}
