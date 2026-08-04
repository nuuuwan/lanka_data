export const DIMENSION_ROW_SX = {
  alignItems: "center",
  display: "grid",
  gap: 0.75,
  gridTemplateColumns: {
    xs: "minmax(0, 1fr) auto",
    sm: "minmax(10rem, 2fr) minmax(7rem, 1fr) minmax(10rem, 2fr) auto",
  },
  mb: 0.75,
};

export const PARENT_DIMENSION_SX = {
  display: "grid",
  gap: 0.75,
  gridTemplateColumns: {
    xs: "1fr",
    sm: "minmax(8rem, 1fr) minmax(5rem, auto) minmax(8rem, 1fr)",
  },
};
