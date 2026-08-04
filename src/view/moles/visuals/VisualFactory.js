import Blocks from "./blocks/Blocks.js";
import AreaBump from "./AreaBump.js";
import BarChart from "./BarChart.js";
import Cartogram from "../../organisms/visuals/Cartogram.js";
import JSONVisual from "./JSONVisual.js";
import Map from "../../organisms/visuals/Map.js";
import HexMap from "../../organisms/visuals/HexMap.js";
import MarimekkoChart from "./MarimekkoChart.js";
import PieChart from "./PieChart.js";
import StackedBarChart from "./StackedBarChart.js";
import SquareMap from "../../organisms/visuals/SquareMap.js";
import UnitHexMap from "../../organisms/visuals/UnitHexMap.js";
import UnitSquareMap from "../../organisms/visuals/UnitSquareMap.js";
import TreeMap from "./TreeMap.js";
import TableVisual from "../../organisms/TableVisual.js";

class VisualFactoryContentsMixin {
  static Blocks = Blocks;
  static AreaBump = AreaBump;
  static BarChart = BarChart;
  static StackedBarChart = StackedBarChart;
  static MarimekkoChart = MarimekkoChart;
  static TreeMap = TreeMap;
  static PieChart = PieChart;
  static Map = Map;
  static HexMap = HexMap;
  static UnitHexMap = UnitHexMap;
  static SquareMap = SquareMap;
  static UnitSquareMap = UnitSquareMap;
  static Cartogram = Cartogram;
  static Table = TableVisual;
  static JSON = JSONVisual;
}

export default class VisualFactory {
  static list() {
    return Object.keys(VisualFactoryContentsMixin);
  }

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
