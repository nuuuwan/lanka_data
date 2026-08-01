import { createTheme } from "@mui/material/styles";

export const FONT_FAMILY = "Fira Sans,  sans-serif";

export const AppTheme = createTheme({
  palette: {
    primary: {
      main: "#444444",
      light: "#888888",
      dark: "#000000",
      contrastText: "ffffff",
    },
    secondary: {
      main: "#8d153a",
      light: "#ba7288",
      dark: "#540c22",
      contrastText: "#ffffff",
    },
  },
  typography: { fontFamily: FONT_FAMILY },
});
