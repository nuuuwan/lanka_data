import { sortAreaBumpXAxis, toAreaBumpData } from "./AreaBump.js";

test("sorts the x-axis to reduce mixing between adjacent rankings", () => {
  const data = [
    { id: "A", Alpha: 3, Beta: 2, Gamma: 1 },
    { id: "B", Alpha: 1, Beta: 2, Gamma: 3 },
    { id: "C", Alpha: 3, Beta: 1, Gamma: 2 },
  ];

  expect(sortAreaBumpXAxis(data).map(({ id }) => id)).toEqual(["A", "C", "B"]);
  expect(toAreaBumpData(data)[0].data.map(({ x }) => x)).toEqual([
    "A",
    "C",
    "B",
  ]);
});

test("keeps the original order when rankings have equal mixing", () => {
  const data = [
    { id: "A", Alpha: 2, Beta: 1 },
    { id: "B", Alpha: 4, Beta: 2 },
    { id: "C", Alpha: 6, Beta: 3 },
  ];

  expect(sortAreaBumpXAxis(data)).toEqual(data);
});

test("handles empty data without mutating the input", () => {
  const empty = [];
  const data = [
    { id: "A", Alpha: 2, Beta: 1 },
    { id: "B", Alpha: 1, Beta: 2 },
  ];
  const original = JSON.parse(JSON.stringify(data));

  expect(sortAreaBumpXAxis(empty)).toEqual([]);
  sortAreaBumpXAxis(data);
  expect(data).toEqual(original);
});
