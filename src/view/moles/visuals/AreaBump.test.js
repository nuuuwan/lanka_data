import AreaBump, { toAreaBumpData } from "./AreaBump.js";
import VisualFactory from "./VisualFactory.js";

describe("AreaBump", () => {
  test("converts stacked chart rows into Nivo series", () => {
    const data = [
      {
        id: "2020",
        Alpha: 10,
        AlphaColor: "#111111",
        Beta: 20,
        BetaColor: "#222222",
        _barWidth: 30,
      },
      {
        id: "2021",
        Alpha: 15,
        AlphaColor: "#111111",
        Beta: 25,
        BetaColor: "#222222",
        _barWidth: 40,
      },
    ];

    expect(toAreaBumpData(data)).toEqual([
      {
        id: "Alpha",
        color: "#111111",
        data: [
          { x: "2020", y: 10 },
          { x: "2021", y: 15 },
        ],
      },
      {
        id: "Beta",
        color: "#222222",
        data: [
          { x: "2020", y: 20 },
          { x: "2021", y: 25 },
        ],
      },
    ]);
  });

  test("fills missing series values with zero", () => {
    expect(toAreaBumpData([{ id: "2020", Alpha: 10 }, { id: "2021" }])).toEqual(
      [
        {
          id: "Alpha",
          color: undefined,
          data: [
            { x: "2020", y: 10 },
            { x: "2021", y: 0 },
          ],
        },
      ],
    );
  });

  test("is registered as a stacked chart", () => {
    expect(VisualFactory.get("AreaBump")).toBe(AreaBump);
    expect(AreaBump.IS_CHART).toBe(true);
    expect(AreaBump.IS_STACKED).toBe(true);
  });
});
