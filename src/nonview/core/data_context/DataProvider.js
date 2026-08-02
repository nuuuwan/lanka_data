import { useState, useEffect, useMemo } from "react";
import DataContext from "./DataContext.js";
import WWW from "../../base/WWW.js";
import Region from "../thing/concept/category_concept/region/region/Region.js";
import RegionFactory from "../thing/concept/category_concept/region/RegionFactory.js";

async function loadRegionData() {
  const regionData = {};
  await Promise.all(
    RegionFactory.list().map(async (RegionClass) => {
      const classId = RegionClass.regionClassId();
      const url =
        "https://raw.githubusercontent.com" +
        "/nuuuwan/lk_admin_regions/refs/heads/main" +
        `/data/ents/${classId}s.json`;
      regionData[classId] = await WWW.json(url);
    }),
  );
  return regionData;
}

export default function DataProvider({ children }) {
  const [regionData, setRegionData] = useState(null);

  useEffect(() => {
    let cancelled = false;
    loadRegionData().then((data) => {
      if (!cancelled) {
        Region.load(data);
        setRegionData(data);
      }
    });
    return () => {
      cancelled = true;
    };
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
