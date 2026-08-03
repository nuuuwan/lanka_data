import { createTheme } from "@mui/material/styles";

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
