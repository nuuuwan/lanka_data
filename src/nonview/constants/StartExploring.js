export const START_EXPLORING_STEPS = [
  {
    question: "How does religion vary across Sri Lanka's provinces?",
    interpretation:
      "The 2024 census view highlights regional differences in religious affiliation.",
    query: "Person/Time=2024+Province+Religion/Count/Blocks",
  },
  {
    question: "How did party support differ by district in the 1994 election?",
    interpretation:
      "The stacked bars make the geographic balance of party support easy to compare.",
    query: "Vote/ElectionType+Time=1994+ED+Party/Count/StackedBarChart",
  },
  {
    question:
      "What did the 2005 presidential result look like across Sri Lanka?",
    interpretation:
      "The cartogram emphasizes electoral districts according to their result totals.",
    query: "Vote/ElectionType=presidential+Time=2005+ED+Party/Count/Cartogram",
  },
];
