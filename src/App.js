import { BrowserRouter, Navigate, Routes, Route } from "react-router-dom";
import QueryPage from "./view/pages/QueryPage";
import { ThemeProvider } from "@mui/material/styles";
import { AppTheme } from "./AppTheme";

const DEFAULT_QUERY_STR = "/Person/Time+Country+Religion/Count";

function App() {
  return (
    <ThemeProvider theme={AppTheme}>
      <BrowserRouter basename="/lanka_data">
        <div className="App">
          <Routes>
            <Route path="*" element={<QueryPage />} />
            <Route path="" element={<Navigate to={DEFAULT_QUERY_STR} />} />
          </Routes>
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
