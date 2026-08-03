import AbstractDataSource from "./AbstractDataSource.js";
import {
  CENSUS_2012_DATASET_TITLE,
  CENSUS_SOURCE,
  ELECTION_DATASET_TITLE,
  ELECTION_SOURCE,
} from "../../constants/DATA_PROVENANCE.js";

export default class GIG extends AbstractDataSource {
  static getProvenanceForQuery(query) {
    const isElectionQuery = query.entityClass.name === "Vote";
    const dataSource = isElectionQuery ? ELECTION_SOURCE : CENSUS_SOURCE;
    return {
      source: dataSource.name,
      title: isElectionQuery
        ? ELECTION_DATASET_TITLE
        : CENSUS_2012_DATASET_TITLE,
      url: dataSource.url,
    };
  }

  static getBaseURL() {
    return (
      "https://raw.githubusercontent.com" +
      "/nuuuwan/gig-data" +
      "/refs/heads/master"
    );
  }
  static getMetadataURL() {
    return this.getBaseURL() + "/lanka_data/metadata.json";
  }
}
