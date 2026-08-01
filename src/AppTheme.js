import { createTheme } from "@mui/material/styles";

export const FONT_FAMILY = "Fira Sans,  sans-serif";

export const AppTheme = createTheme({
  palette: {
    primary: {
      main: "#444444",
      light: "#888888",
      dark: "#000000",
      contrastText: "#ffffff",
    },
    info: {
      main: "#00534E",
      light: "#669794",
      dark: "#00312E",
      contrastText: "#ffffff",
    },
  },
  typography: { fontFamily: FONT_FAMILY },
});
