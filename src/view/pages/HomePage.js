import { Typography } from "@mui/material";
import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import useDataContext from "../../nonview/core/data_context/useDataContext.js";
import ExampleQueryLinks from "../moles/ExampleQueryLinks.js";
import QueryMenuAppBar from "../organisms/QueryMenuAppBar.js";
import VisualQueryForm from "../organisms/VisualQueryForm.js";
import styles from "./HomePage.module.css";

export default function HomePage() {
  const [input, setInput] = useState("");
  const navigate = useNavigate();
  const location = useLocation();
  const { isReady, queryOptions } = useDataContext();

  useEffect(() => {
    if (location.search && location.search[1] === "/") {
      const decoded = location.search
        .slice(1)
        .split("&")
        .map((s) => s.replace(/~and~/g, "&"))
        .join("?");
      if (decoded && decoded !== "/") {
        navigate(decoded, { replace: true });
      }
    }
  }, [location.search, navigate]);

  const submit = () => {
    const query = input.trim();
    if (query) navigate(`/${query}`);
  };

  return (
    <>
      <QueryMenuAppBar />
      <main className={styles.page}>
        <Typography component="h2" variant="h5">
          Explore Sri Lanka through data
        </Typography>
        <Typography className={styles.introduction}>
          One query, one tool for people and code. Start with an example to
          explore census and election data through charts, maps, and tables.
        </Typography>
        <ExampleQueryLinks />
      </main>
      <VisualQueryForm
        value={input}
        onChange={setInput}
        onSubmit={submit}
        queryOptions={queryOptions}
        disabled={!isReady}
      />
    </>
  );
}
