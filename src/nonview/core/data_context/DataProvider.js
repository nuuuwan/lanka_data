import { useState, useEffect, useMemo } from "react";
import DataContext from "./DataContext.js";
import RegionFactory from "../thing/concept/category_concept/region/RegionFactory.js";

async function loadRegionData() {
  const regionData = {};
  await Promise.all(
    RegionFactory.list().map(async (RegionClass) => {
      const classId = RegionClass.regionClassId();
      const ents = await RegionClass.loadEnts();
      regionData[classId] = ents;
      RegionClass.ents = ents;
    }),
  );
  return regionData;
}

export default function DataProvider({ children }) {
  const [regionData, setRegionData] = useState(null);

  useEffect(() => {
    loadRegionData().then((data) => {
      setRegionData(data);
    });
  }, []);

  const value = useMemo(
    () => ({
      isReady: regionData !== null,
      regionData,
    }),
    [regionData],
  );

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}
