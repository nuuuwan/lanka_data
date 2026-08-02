import { useEffect, useState } from "react";
import { feature } from "topojson-client";

import WWW from "./WWW.js";

export default function useGeoJson(regionClass) {
  const [geoJson, setGeoJson] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const topoJson = await WWW.json(regionClass.getGeoURL());
      if (!cancelled) {
        setGeoJson(feature(topoJson, topoJson.objects.data));
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [regionClass]);

  return geoJson;
}
