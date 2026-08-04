import { useEffect, useState } from "react";
import { feature } from "topojson-client";

import WWW from "./WWW.js";

const geoJsonByURL = new Map();
const geoJsonRequestByURL = new Map();

async function loadGeoJson(geoURL) {
  const cachedGeoJson = geoJsonByURL.get(geoURL);
  if (cachedGeoJson) {
    return cachedGeoJson;
  }

  if (!geoJsonRequestByURL.has(geoURL)) {
    const request = WWW.json(geoURL)
      .then((topoJson) => feature(topoJson, topoJson.objects.data))
      .then((geoJson) => {
        geoJsonByURL.set(geoURL, geoJson);
        return geoJson;
      })
      .finally(() => geoJsonRequestByURL.delete(geoURL));
    geoJsonRequestByURL.set(geoURL, request);
  }

  return geoJsonRequestByURL.get(geoURL);
}

export default function useGeoJson(regionClass) {
  const geoURL = regionClass.getGeoURL();
  const [geoJson, setGeoJson] = useState(() => geoJsonByURL.get(geoURL));

  useEffect(() => {
    let cancelled = false;
    const cachedGeoJson = geoJsonByURL.get(geoURL);
    if (cachedGeoJson) {
      setGeoJson(cachedGeoJson);
      return undefined;
    }

    setGeoJson(null);
    async function load() {
      const nextGeoJson = await loadGeoJson(geoURL);
      if (!cancelled) {
        setGeoJson(nextGeoJson);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [geoURL]);

  return geoJson;
}
