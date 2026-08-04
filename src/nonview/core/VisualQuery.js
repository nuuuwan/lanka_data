import Query from "./Query.js";

export default class VisualQuery {
  constructor(query, visualClassName, visualQueryStr) {
    this.query = query;
    this.visualClassName = visualClassName;
    this.visualQueryStr = visualQueryStr;
  }

  toString() {
    return this.visualQueryStr;
  }

  static async fromString(visualQueryStr) {
    const tokens = visualQueryStr.split(Query.DELIM_TOKEN).filter(Boolean);
    if (tokens.length !== 4) {
      throw new Error(
        "Use the format Entity/Dimensions/Aggregate/Visualization. " +
          "For example: Vote/ElectionType=presidential+Time=2024+PD/Count/Map.",
      );
    }
    const visualClassName = tokens[tokens.length - 1];
    const queryStr = tokens.slice(0, tokens.length - 1).join(Query.DELIM_TOKEN);
    const query = await Query.fromString(queryStr);
    return new VisualQuery(query, visualClassName, visualQueryStr);
  }
}
