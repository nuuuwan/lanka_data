import { Typography } from "@mui/material";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import useDataContext from "../../nonview/core/data_context/useDataContext.js";
import ExampleQueryLinks from "../moles/ExampleQueryLinks.js";
import QueryMenuAppBar from "../organisms/QueryMenuAppBar.js";
import VisualQueryForm from "../organisms/VisualQueryForm.js";
import styles from "./HomePage.module.css";

export default function HomePage() {
  const [input, setInput] = useState("");
  const navigate = useNavigate();
  const { isReady, queryOptions } = useDataContext();
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
