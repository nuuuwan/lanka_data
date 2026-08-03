import { createContext } from "react";

const DataContext = createContext({
  isReady: false,
  queryOptions: {
    entities: [],
    dimensionsByEntity: {},
  },
});

export default DataContext;
