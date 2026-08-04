import {
  getFilterLabel,
  getGeographyLabel,
  getTimePeriodLabel,
} from "./VisualMetadataDetails.js";
import {
  getPopulationLabel,
  getRequestedDimensions,
} from "./VisualMetadataLabels.js";
import { getTitleLabel, getUnitsLabel } from "./VisualMetadataTitle.js";

export default class VisualMetadata {
  static from(query, datumSet) {
    const dimensions = getRequestedDimensions(query);
    const population = getPopulationLabel(query);
    return {
      title: getTitleLabel(query),
      subtitle: [
        `Population: ${population}`,
        `Geography: ${getGeographyLabel(query, dimensions)}`,
        `Time period: ${getTimePeriodLabel(dimensions)}`,
        `Units: ${getUnitsLabel(query, datumSet, population)}`,
        `Filters: ${getFilterLabel(query, dimensions)}`,
      ].join(" • "),
    };
  }
}
