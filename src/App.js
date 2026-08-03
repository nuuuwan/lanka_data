import { BrowserRouter, Navigate, Routes, Route } from "react-router-dom";
import VisualQueryPage from "./view/pages/VisualQueryPage";
import { ThemeProvider } from "@mui/material/styles";
import { AppTheme } from "./AppTheme";
import AppFooter from "./view/atoms/AppFooter.js";
import DataProvider from "./nonview/core/data_context/DataProvider.js";
const DEFAULT_VISUAL_QUERY_STR =
  "/Vote/ElectionType=presidential+Time=2024+PD<ED=colombo+Party/Count/SquareMap";

function App() {
  return (
    <ThemeProvider theme={AppTheme}>
      <DataProvider>
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
      </DataProvider>
    </ThemeProvider>
  );
}

export default App;
