import { useEffect, useRef, useState } from "react";
import { Alert, AlertTitle, Box } from "@mui/material";
import { useNavigate, useParams } from "react-router-dom";
import useDataContext from "../../nonview/core/data_context/useDataContext.js";
import LoadingProgress from "../molecules/LoadingProgress.js";
import VisualQueryForm from "../organisms/VisualQueryForm.js";
import QueryMenuAppBar from "../organisms/QueryMenuAppBar.js";
import VisualErrorBoundary from "../organisms/VisualErrorBoundary.js";
import styles from "./VisualQueryPage.module.css";
import VisualContent from "./VisualContent.js";
import useVisualData from "./useVisualData.js";
import useVisualQuery from "./useVisualQuery.js";

export default function VisualQueryPage() {
  const { "*": queryString } = useParams();
  const navigate = useNavigate();
  const { isReady, queryOptions } = useDataContext();
  const [input, setInput] = useState(queryString);
  const [visualReadyQuery, setVisualReadyQuery] = useState(null);
  const visualTitleRef = useRef(null);
  const parse = useVisualQuery(isReady, queryString);
  const load = useVisualData(parse.visualQuery);
  const VisualClass = parse.visualQuery?.visualClass;
  const errorMessage = parse.errorMessage || load.errorMessage;
  const visualDataReady =
    isReady &&
    Boolean(VisualClass) &&
    load.datumSet !== null &&
    load.loadTimeSeconds !== null;
  const isVisualReady = visualDataReady && visualReadyQuery === queryString;
  const isLoading = !isVisualReady;
  useEffect(() => setInput(queryString), [queryString]);
  useEffect(() => {
    if (!visualDataReady || errorMessage) {
      return undefined;
    }
    const frameId = requestAnimationFrame(() =>
      setVisualReadyQuery(queryString),
    );
    return () => cancelAnimationFrame(frameId);
  }, [errorMessage, queryString, visualDataReady]);
  useEffect(() => {
    if (isLoading || errorMessage || !visualTitleRef.current) return;
    visualTitleRef.current.scrollIntoView?.({
      behavior: "smooth",
      block: "start",
    });
  }, [errorMessage, isLoading]);
  const loadedQuery =
    load.datumSet?.datumList.length > 0 && load.loadTimeSeconds !== null
      ? queryString
      : null;
  const submit = (valueToSubmit = input) => {
    const nextQuery = valueToSubmit.trim();
    if (nextQuery && nextQuery !== queryString) navigate(`/${nextQuery}`);
  };
  return (
    <>
      <QueryMenuAppBar loadedVisualQuery={loadedQuery} />
      <Box className={styles.page}>
        {errorMessage ? (
          <Alert severity="error" data-testid="query-error">
            <AlertTitle>Sorry, something went wrong.</AlertTitle>
            {errorMessage}
          </Alert>
        ) : (
          <Box data-testid={isLoading ? undefined : "visual-content"}>
            {isLoading ? (
              <LoadingProgress />
            ) : (
              <VisualErrorBoundary key={queryString}>
                <VisualContent
                  VisualClass={VisualClass}
                  datumSet={load.datumSet}
                  loadTimeSeconds={load.loadTimeSeconds}
                  query={parse.visualQuery.query}
                  visualTitleRef={visualTitleRef}
                />
              </VisualErrorBoundary>
            )}
          </Box>
        )}
        {isReady && (
          <VisualQueryForm
            value={input}
            onChange={setInput}
            onSubmit={submit}
            queryOptions={queryOptions}
            disabled={isLoading && !errorMessage}
            loadedVisualQuery={loadedQuery}
          />
        )}
      </Box>
    </>
  );
}
