import { useContext } from "react";

import DataContext from "./DataContext.js";

export default function useDataContext() {
  return useContext(DataContext);
}
