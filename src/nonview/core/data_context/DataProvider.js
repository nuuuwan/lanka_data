import { useState, useEffect, useMemo } from "react";
import DataContext from "./DataContext.js";
import DataSourceFactory from "../data_source/DataSourceFactory.js";
import RegionFactory from "../thing/concept/category_concept/region/RegionFactory.js";

async function loadRegionData() {
  const regionData = {};
  console.debug("[DataProvider] Loading region data");
  await Promise.all(
    RegionFactory.list().map(async (RegionClass) => {
      const classId = RegionClass.regionClassId();
      console.debug(`[DataProvider] Loading region class "${classId}"`);
      const ents = await RegionClass.loadEnts();
      regionData[classId] = ents;
      RegionClass.ents = ents;
      console.debug(
        `[DataProvider] Loaded ${ents.length} entities for region class "${classId}"`,
      );
    }),
  );
  console.debug(
    `[DataProvider] Region data ready (${Object.keys(regionData).length} classes)`,
  );
  return regionData;
}

export default function DataProvider({ children }) {
  const [regionData, setRegionData] = useState(null);
  const [queryOptions, setQueryOptions] = useState(null);

  useEffect(() => {
    loadRegionData().then((data) => {
      setRegionData(data);
    });
    DataSourceFactory.getQueryOptions().then(setQueryOptions);
  }, []);

  const value = useMemo(
    () => ({
      isReady: regionData !== null && queryOptions !== null,
      regionData,
      queryOptions,
    }),
    [queryOptions, regionData],
  );

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}
