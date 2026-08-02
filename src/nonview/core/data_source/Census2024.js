import AbstractDataSource from "./AbstractDataSource.js";

export default class Census2024 extends AbstractDataSource {
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
