import Blocks from "./blocks/Blocks.js";
import BarChart from "./BarChart.js";
import Cartogram from "./Cartogram.js";
import JSONVisual from "./JSONVisual.js";
import Map from "./Map.js";
import HexMap from "./HexMap.js";
import MarimekkoChart from "./MarimekkoChart.js";
import StackedBarChart from "./StackedBarChart.js";
import TreeMap from "./TreeMap.js";

class VisualFactoryContentsMixin {
  static Blocks = Blocks;
  static BarChart = BarChart;
  static StackedBarChart = StackedBarChart;
  static MarimekkoChart = MarimekkoChart;
  static TreeMap = TreeMap;
  static Map = Map;
  static HexMap = HexMap;
  static Cartogram = Cartogram;
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
