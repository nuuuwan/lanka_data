const RELATIVE_TIME_UNITS = [
  ["year", 60 * 60 * 24 * 365],
  ["month", 60 * 60 * 24 * 30],
  ["week", 60 * 60 * 24 * 7],
  ["day", 60 * 60 * 24],
  ["hour", 60 * 60],
  ["minute", 60],
  ["second", 1],
];

export default function formatRelativeTime(timestamp, now = Date.now()) {
  const differenceInSeconds = (timestamp - now) / 1000;
  const [unit, secondsPerUnit] =
    RELATIVE_TIME_UNITS.find(
      ([, unitDuration]) =>
        Math.abs(differenceInSeconds) >= unitDuration,
    ) ?? RELATIVE_TIME_UNITS.at(-1);

  return new Intl.RelativeTimeFormat(undefined, {
    numeric: "always",
  }).format(Math.round(differenceInSeconds / secondsPerUnit), unit);
}
