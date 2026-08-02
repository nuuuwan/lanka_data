import WWW from "../../base/WWW.js";
import Datum from "../Datum.js";

export default class AbstractDataSource {
  static getBaseURL() {
    throw new Error(
      'Abstract method. "getBaseURL" must be implemented in subclass.',
    );
  }
  static getMetadataURL() {
    throw new Error(
      'Abstract method. "getMetadataURL" must be implemented in subclass.',
    );
  }
  static async getMetadata() {
    return await WWW.json(this.getMetadataURL());
  }

  static async getMetadataForQuery(query) {
    const metadata = await this.getMetadata();
    return metadata[query.getMetadataKey()] || [];
  }

  static async getDatumListForPartialPath(partialPath) {
    const url = this.getBaseURL() + "/" + partialPath;
    const lankaData = await WWW.json(url);
    return Datum.listFromLankaData(lankaData);
  }

  static async getDatumListForQuery(query) {
    const metadataForQuery = await this.getMetadataForQuery(query);
    console.debug(
      `Found ${metadataForQuery.length} metadata entries` +
        ` for "${query}" in ${this.name}`,
    );
    if (metadataForQuery.length === 0) {
      return [];
    }
    const candidateDatumListList = await Promise.all(
      metadataForQuery.map((partialPath) =>
        this.getDatumListForPartialPath(partialPath),
      ),
    );
    const candidateDatumList = candidateDatumListList.flat();
    console.debug(
      `Found ${candidateDatumList.length} candidate datums` +
        ` for "${query}" in ${this.name}`,
    );
    const filteredDatumList = candidateDatumList.filter((datum) =>
      datum.query.isSubsetOf(query),
    );
    console.debug(
      `Found ${filteredDatumList.length} filtered datums` +
        ` for "${query}" in ${this.name}`,
    );
    return filteredDatumList;
  }
}
