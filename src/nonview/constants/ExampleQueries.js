export const EXAMPLE_QUERIES = [
  {
    label: "2024 census by province",
    description: "Explore population and religion counts as blocks.",
    query: "Person/Time=2024+Province+Religion/Count/Blocks",
  },
  {
    label: "Religions in Colombo",
    description: "Compare Colombo district religion counts in a bar chart.",
    query: "Person/Time=2024+District=colombo+Religion/Count/BarChart",
  },
  {
    label: "Colombo religion share",
    description: "View Colombo district religion counts in a pie chart.",
    query: "Person/Time=2024+District=colombo+Religion/Count/PieChart",
  },
  {
    label: "1994 election by district",
    description: "Compare party results by electoral district.",
    query: "Vote/ElectionType+Time=1994+ED+Party/Count/StackedBarChart",
  },
  {
    label: "2024 Colombo election map",
    description: "Map presidential party results across Colombo.",
    query:
      "Vote/ElectionType=presidential+Time=2024+PD<ED=colombo+Party/Count/SquareMap",
  },
  {
    label: "2005 presidential cartogram",
    description: "View party results by electoral district on a cartogram.",
    query: "Vote/ElectionType=presidential+Time=2005+ED+Party/Count/Cartogram",
  },
];
