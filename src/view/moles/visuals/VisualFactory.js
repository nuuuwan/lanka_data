import Blocks from "./blocks/Blocks.js";
import JSONVisual from "./JSONVisual.js";

class VisualFactoryContentsMixin {
  static Blocks = Blocks;
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
