import AbstractDataSource from "./AbstractDataSource.js";

export default class GIG extends AbstractDataSource {
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
