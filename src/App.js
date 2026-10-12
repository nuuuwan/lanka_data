import { BrowserRouter, Routes, Route } from "react-router-dom";
import VisualQueryPage from "./view/pages/VisualQueryPage";
import { ThemeProvider } from "@mui/material/styles";
import { AppTheme } from "./AppTheme";
import DataProvider from "./nonview/core/data_context/DataProvider.js";
import { APP_PATH } from "./nonview/constants/APP.js";

function App() {
  return (
    <ThemeProvider theme={AppTheme}>
      <DataProvider>
        <BrowserRouter basename={APP_PATH}>
          <Routes>
            <Route path="*" element={<VisualQueryPage />} />
          </Routes>
        </BrowserRouter>
      </DataProvider>
    </ThemeProvider>
  );
}

export default App;
