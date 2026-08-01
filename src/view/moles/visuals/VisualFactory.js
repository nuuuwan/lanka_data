import Blocks from "./blocks/Blocks.js";
import BarChart from "./BarChart.js";
import JSONVisual from "./JSONVisual.js";
import StackedBarChart from "./StackedBarChart.js";

class VisualFactoryContentsMixin {
  static Blocks = Blocks;
  static BarChart = BarChart;
  static StackedBarChart = StackedBarChart;
  static JSON = JSONVisual;
}

export default class VisualFactory {
  static get(visualType) {
    const VisualClass = VisualFactoryContentsMixin[visualType];
    if (!VisualClass) {
      throw new Error(
        `Visual type "${visualType}" not found in VisualFactory.`,
      );
    }
    return VisualClass;
  }
}
