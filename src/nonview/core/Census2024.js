import WWW from "../../nonview/base/WWW.js";
import DatumSet from "./DatumSet.js";
import Datum from "./Datum.js";

export default class Census2024 {
  static URL_REPO = "https://raw.githubusercontent.com/nuuuwan/lk_census_2024";
  static URL_BASE = Census2024.URL_REPO + "/refs/heads/main";

  static URL_METADATA =
    Census2024.URL_BASE + "/metadata/lanka_data.metadata.json";

  static async getMetadata() {
    return await WWW.json(Census2024.URL_METADATA);
  }

  static async getMetadataForQuery(query) {
    const metadata = await Census2024.getMetadata();
    return metadata[query.toString()] || [];
  }

  static async getDatumListForPartialPath(partialPath) {
    const url = Census2024.URL_BASE + "/" + partialPath;
    const lankaData = await WWW.json(url);
    return Datum.listFromLankaData(lankaData);
  }

  static async getDatumSetForQuery(query) {
    const metadataForQuery = await Census2024.getMetadataForQuery(query);
    if (metadataForQuery.length === 0) {
      return new DatumSet([]);
    }
    const partialPath = metadataForQuery[0];
    const candidateDatumList =
      await Census2024.getDatumListForPartialPath(partialPath);
    const datumList = candidateDatumList.filter((datum) =>
      datum.query.isSubsetOf(query),
    );
    return new DatumSet(datumList);
  }
}
