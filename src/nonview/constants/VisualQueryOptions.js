export const DIMENSION_OPERATORS = [
  { value: "", label: "None" },
  { value: "=", label: "=" },
  { value: ":", label: ":" },
  { value: "<", label: "<" },
];

export const VISUAL_GROUPS = [
  {
    label: "Charts",
    visuals: [
      "AreaBump",
      "BarChart",
      "StackedBarChart",
      "MarimekkoChart",
      "PieChart",
      "TreeMap",
    ],
  },
  {
    label: "Maps",
    visuals: [
      "Map",
      "Cartogram",
      "HexMap",
      "UnitHexMap",
      "SquareMap",
      "UnitSquareMap",
    ],
  },
  {
    label: "Other",
    visuals: ["Blocks", "Table", "JSON"],
  },
];
