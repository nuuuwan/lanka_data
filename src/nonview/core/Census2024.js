import WWW from "../../nonview/base/WWW.js";

export default class Census2024 {
  static URL_REPO = "https://raw.githubusercontent.com/nuuuwan/lk_census_2024";
  static URL_BASE = Census2024.URL_REPO + "/refs/heads/main";

  static URL_METADATA =
    Census2024.URL_BASE + "/metadata/lanka_data.metadata.json";

  static async getMetadata() {
    return await WWW.json(Census2024.URL_METADATA);
  }

  static async getMetadataForQuery(queryStr) {
    const metadata = await Census2024.getMetadata();
    return metadata[queryStr] || [];
  }

  static async getLankaDataForPartialPath(partialPath) {
    const url = Census2024.URL_BASE + "/" + partialPath;
    return await WWW.json(url);
  }

  static async getLankaDataForQuery(queryStr) {
    const metadataForQuery = await Census2024.getMetadataForQuery(queryStr);
    if (metadataForQuery.length === 0) {
      return {};
    }
    const partialPath = metadataForQuery[0];
    return await Census2024.getLankaDataForPartialPath(partialPath);
  }
}
