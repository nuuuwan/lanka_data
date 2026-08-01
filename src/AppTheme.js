import { createTheme } from "@mui/material/styles";

export const PRIMARY_COLOR = "#0b6e4f";

export const FONT_FAMILY = "Fira Sans,  sans-serif";

export const AppTheme = createTheme({
  palette: { mode: "light", primary: { main: PRIMARY_COLOR } },
  typography: { fontFamily: FONT_FAMILY },
});
