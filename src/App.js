import { BrowserRouter, Navigate, Routes, Route } from "react-router-dom";
import VisualQueryPage from "./view/pages/VisualQueryPage";
import { ThemeProvider } from "@mui/material/styles";
import { AppTheme } from "./AppTheme";
import AppFooter from "./view/atoms/AppFooter.js";
const DEFAULT_VISUAL_QUERY_STR =
  "/Vote/ElectionType=presidential+Time=2024+Province+Party/Count/MarimekkoChart";

function App() {
  return (
    <ThemeProvider theme={AppTheme}>
      <BrowserRouter basename="/lanka_data">
        <div className="App">
          <Routes>
            <Route path="*" element={<VisualQueryPage />} />
            <Route
              path=""
              element={<Navigate to={DEFAULT_VISUAL_QUERY_STR} />}
            />
          </Routes>
        </div>
      </BrowserRouter>
      <AppFooter />
    </ThemeProvider>
  );
}

export default App;
