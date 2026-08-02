import { createTheme } from "@mui/material/styles";

export const FONT_FAMILY = "Fira Code,  sans-serif";

export const AppTheme = createTheme({
  palette: {
    primary: {
      main: "#444444",
      light: "#CCDCDB",
      dark: "#000000",
      contrastText: "#ffffff",
    },
    info: {
      main: "#669794",
      light: "#CCDCDB",
      dark: "#00534E",
      contrastText: "#ffffff",
    },
  },
  typography: { fontFamily: FONT_FAMILY },
});
