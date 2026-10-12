import { BrowserRouter, Routes, Route } from "react-router-dom";
import VisualQueryPage from "./view/pages/VisualQueryPage";
import HomePage from "./view/pages/HomePage.js";
import { ThemeProvider } from "@mui/material/styles";
import { AppTheme } from "./AppTheme";
import DataProvider from "./nonview/core/data_context/DataProvider.js";
import { APP_PATH } from "./nonview/constants/APP.js";

function App() {
  return (
    <ThemeProvider theme={AppTheme}>
      <BrowserRouter basename={APP_PATH}>
        <DataProvider>
          <Routes>
            <Route index element={<HomePage />} />
            <Route path="*" element={<VisualQueryPage />} />
          </Routes>
        </DataProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
