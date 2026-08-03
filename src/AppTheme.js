import { createTheme } from "@mui/material/styles";

export const BODY_FONT_FAMILY =
  '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif';
export const HEADING_FONT_FAMILY = 'Georgia, "Times New Roman", Times, serif';
export const FONT_FAMILY = BODY_FONT_FAMILY;

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
  typography: {
    fontFamily: BODY_FONT_FAMILY,
    h1: { fontFamily: HEADING_FONT_FAMILY },
    h2: { fontFamily: HEADING_FONT_FAMILY },
    h3: { fontFamily: HEADING_FONT_FAMILY },
    h4: { fontFamily: HEADING_FONT_FAMILY },
    h5: { fontFamily: HEADING_FONT_FAMILY },
    h6: { fontFamily: HEADING_FONT_FAMILY },
    subtitle1: { fontFamily: HEADING_FONT_FAMILY },
    subtitle2: { fontFamily: HEADING_FONT_FAMILY },
  },
  components: {
    MuiAlert: {
      styleOverrides: {
        message: {
          maxWidth: "65ch",
        },
      },
    },
    MuiFormHelperText: {
      styleOverrides: {
        root: {
          maxWidth: "65ch",
        },
      },
    },
  },
});
