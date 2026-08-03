import AbstractDataSource from "./AbstractDataSource.js";
import {
  CENSUS_2024_DATASET_TITLE,
  CENSUS_SOURCE,
} from "../../constants/DATA_PROVENANCE.js";

export default class Census2024 extends AbstractDataSource {
  static getProvenanceForQuery() {
    return {
      source: CENSUS_SOURCE.name,
      title: CENSUS_2024_DATASET_TITLE,
      url: CENSUS_SOURCE.url,
    };
  }

  static getBaseURL() {
    return (
      "https://raw.githubusercontent.com" +
      "/nuuuwan/lk_census_2024" +
      "/refs/heads/main"
    );
  }
  static getMetadataURL() {
    return this.getBaseURL() + "/metadata/lanka_data.metadata.json";
  }
}
