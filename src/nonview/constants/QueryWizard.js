export const QUERY_WIZARD_STEPS = ["Data", "Facets", "Visual"];
export const QUERY_WIZARD_HELP = [
  {
    introduction: "Choose what to explore. A few examples:",
    examples: ["Person: population data.", "Vote: election results."],
    tip: "The wizard updates the query as you edit. To apply it, close the wizard and use Refresh or press Enter in the query bar.",
  },
  {
    introduction:
      "Each row groups or filters your data. Here are a few examples; explore the dropdowns for more fields, operators, and values:",
    examples: [
      "Province, Operator None: group by province.",
      "Time = 2024: filter to 2024.",
      "Party, Operator None: compare parties.",
      "PD < ED = colombo: polling divisions in Colombo electoral district.",
    ],
  },
  {
    introduction:
      "Choose a display for your question. A few of the available options:",
    examples: [
      "Bar Chart: compare parties or provinces.",
      "Pie Chart: show their shares.",
      "Table: list exact values.",
    ],
    tip: "Close the wizard, then use Refresh or press Enter in the query bar to update the visual.",
  },
];
