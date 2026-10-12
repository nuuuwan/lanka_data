export const EXAMPLE_QUERIES = [
  {
    label: "2024 Colombo religions: bar chart",
    image: "colombo-religion-bar-chart.png",
    query: "Person/Time=2024+District=colombo+Religion/Count/BarChart",
  },
  {
    label: "2024 Colombo religion shares: pie chart",
    image: "colombo-religion-pie-chart.png",
    query: "Person/Time=2024+District=colombo+Religion/Count/PieChart",
  },
  {
    label: "1994 election votes by district and party",
    image: "1994-election-stacked-bar-chart.png",
    query: "Vote/ElectionType+Time=1994+ED+Party/Count/StackedBarChart",
  },
  {
    label: "2024 Colombo election map",
    image: "2024-colombo-election-square-map.png",
    query:
      "Vote/ElectionType=presidential+Time=2024+PD<ED=colombo+Party/Count/SquareMap",
  },
  {
    label: "2005 presidential cartogram",
    image: "2005-presidential-cartogram.png",
    query: "Vote/ElectionType=presidential+Time=2005+ED+Party/Count/Cartogram",
  },
  {
    label: "2024 presidential hex map",
    image: "2024-presidential-hex-map.png",
    query: "Vote/ElectionType=presidential+Time=2024+ED+Party/Count/HexMap",
  },
  {
    label: "2024 province religions: unit square map",
    image: "2024-religion-unit-square-map.png",
    query: "Person/Time=2024+Province+Religion/Count/UnitSquareMap",
  },
  {
    label: "2024 province populations and religions: Marimekko",
    image: "2024-religion-marimekko-chart.png",
    query: "Person/Time=2024+Province+Religion/Count/MarimekkoChart",
  },
  {
    label: "2024 province religions: unit hex map",
    image: "2024-religion-unit-hex-map.png",
    query: "Person/Time=2024+Province+Religion/Count/UnitHexMap",
  },
];
