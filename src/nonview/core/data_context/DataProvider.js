import { useState, useEffect, useMemo } from "react";
import DataContext from "./DataContext.js";
import WWW from "../../base/WWW.js";
import Region from "../thing/concept/category_concept/region/region/Region.js";

const REGION_CLASS_IDS = [
  "country",
  "province",
  "district",
  "dsd",
  "ed",
  "gnd",
  "pd",
];

async function loadRegionData() {
  const regionData = {};
  await Promise.all(
    REGION_CLASS_IDS.map(async (classId) => {
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
