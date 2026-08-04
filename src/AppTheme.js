import { createTheme } from "@mui/material/styles";

export const APP_FONT_FAMILY = '"Fira Code", monospace';

export const NIVO_THEME = {
  text: {
    fontFamily: APP_FONT_FAMILY,
  },
  axis: {
    legend: {
      text: {
        fontSize: 16,
      },
    },
    ticks: {
      text: {
        fontSize: 14,
      },
    },
  },
};

export const AppTheme = createTheme({
  typography: {
    fontFamily: APP_FONT_FAMILY,
  },
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
