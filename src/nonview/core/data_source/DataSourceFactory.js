import Census2024 from "./Census2024.js";
import GIG from "./GIG.js";
import DatumSet from "../DatumSet.js";

export default class DataSourceFactory {
  static getDataSourceClasses() {
    return [Census2024, GIG];
  }
  static async getDatumSetForQuery(query) {
    const datumListList = await Promise.all(
      this.getDataSourceClasses().map((dataSourceClass) =>
        dataSourceClass.getDatumListForQuery(query),
      ),
    );
    const datumList = datumListList.flat();
    console.debug(`Found ${datumList.length} datums for "${query}"`);

    return new DatumSet(datumList);
  }
}
