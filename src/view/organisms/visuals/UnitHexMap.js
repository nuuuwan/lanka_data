import HexMap from "./HexMap.js";

export default function UnitHexMap({ datumSet }) {
  return <HexMap datumSet={datumSet} isUnit />;
}

UnitHexMap.IS_CHART = false;
