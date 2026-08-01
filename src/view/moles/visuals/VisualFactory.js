import Blocks from "./blocks/Blocks.js";

class VisualFactoryContentsMixin {
  static Blocks = Blocks;
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
