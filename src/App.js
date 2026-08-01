import { BrowserRouter, Navigate, Routes, Route } from "react-router-dom";
import QueryPage from "./view/pages/QueryPage";

const DEFAULT_QUERY_STR = "/Person/Time*Country*Religion/Count";

function App() {
  return (
    <BrowserRouter basename="/lanka_data">
      <div className="App">
        <Routes>
          <Route path="*" element={<QueryPage />} />
          <Route path="" element={<Navigate to={DEFAULT_QUERY_STR} />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
