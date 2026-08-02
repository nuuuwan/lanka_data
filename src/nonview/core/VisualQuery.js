import Query from "./Query.js";
import VisualFactory from "../../view/moles/visuals/VisualFactory.js";

export default class VisualQuery {
  constructor(query, visualClass, visualQueryStr) {
    this.query = query;
    this.visualClass = visualClass;
    this.visualQueryStr = visualQueryStr;
  }

  toString() {
    return this.visualQueryStr;
  }

  static async fromString(visualQueryStr) {
    const tokens = visualQueryStr.split(Query.DELIM_TOKEN);
    const visualClassName = tokens[tokens.length - 1];
    const visualClass = VisualFactory.get(visualClassName);
    const queryStr = tokens.slice(0, tokens.length - 1).join(Query.DELIM_TOKEN);
    const query = await Query.fromString(queryStr);
    return new VisualQuery(query, visualClass, visualQueryStr);
  }
}
