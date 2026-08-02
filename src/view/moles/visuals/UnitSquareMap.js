import SquareMap from "./SquareMap.js";

export default function UnitSquareMap({ datumSet }) {
  return <SquareMap datumSet={datumSet} isUnit />;
}

UnitSquareMap.IS_CHART = false;
