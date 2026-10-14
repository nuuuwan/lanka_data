import DATA_SOURCES from "../../constants/DataSources.json";
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
    return DATA_SOURCES.Census2024.baseURL;
  }
  static getMetadataURL() {
    return this.getBaseURL() + "/" + DATA_SOURCES.Census2024.metadataPath;
  }
}
