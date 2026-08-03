import Census2024 from "./Census2024.js";
import GIG from "./GIG.js";
import DatumSet from "../DatumSet.js";

export default class DataSourceFactory {
  static getDataSourceClasses() {
    return [Census2024, GIG];
  }

  static getQueryOptionsFromMetadata(metadataList) {
    const entities = new Set();
    const dimensionsByEntity = {};

    metadataList.forEach((metadata) => {
      Object.keys(metadata).forEach((metadataKey) => {
        const [entity, dimensions = ""] = metadataKey.split("/");
        entities.add(entity);
        dimensionsByEntity[entity] ??= new Set();
        dimensions
          .split("+")
          .filter(Boolean)
          .forEach((dimension) => dimensionsByEntity[entity].add(dimension));
      });
    });

    return {
      entities: [...entities].sort(),
      dimensionsByEntity: Object.fromEntries(
        Object.entries(dimensionsByEntity).map(([entity, dimensions]) => [
          entity,
          [...dimensions].sort(),
        ]),
      ),
    };
  }

  static async getQueryOptions() {
    const metadataList = await Promise.all(
      this.getDataSourceClasses().map((dataSourceClass) =>
        dataSourceClass.getMetadata(),
      ),
    );
    return this.getQueryOptionsFromMetadata(metadataList);
  }

  static async getDatumSetForQuery(query) {
    console.debug(`[DataSourceFactory] Loading datums for "${query}"`);
    const datumListList = await Promise.all(
      this.getDataSourceClasses().map(async (dataSourceClass) => {
        console.debug(
          `[DataSourceFactory] Querying ${dataSourceClass.name} for "${query}"`,
        );
        const datumList = await dataSourceClass.getDatumListForQuery(query);
        console.debug(
          `[DataSourceFactory] ${dataSourceClass.name} returned ${datumList.length} datums`,
        );
        return datumList;
      }),
    );
    const datumList = datumListList.flat();
    console.debug(
      `[DataSourceFactory] Loaded ${datumList.length} total datums for "${query}"`,
    );

    return new DatumSet(datumList);
  }
}
