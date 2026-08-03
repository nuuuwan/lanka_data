import { useContext, useEffect, useRef, useState } from "react";
import { Alert, AlertTitle, Box } from "@mui/material";
import { useNavigate, useParams } from "react-router-dom";
import DataContext from "../../nonview/core/data_context/DataContext.js";
import LoadingProgress from "../molecules/LoadingProgress.js";
import ChangeViewSection from "../organisms/ChangeViewSection.js";
import QueryMenuAppBar from "../organisms/QueryMenuAppBar.js";
import VisualErrorBoundary from "../organisms/VisualErrorBoundary.js";
import styles from "./VisualQueryPage.module.css";
import VisualContent from "./VisualContent.js";
import useApplicationLoad from "./useApplicationLoad.js";
import useLoadingSteps from "./useLoadingSteps.js";
import useVisualData from "./useVisualData.js";
import useVisualQuery from "./useVisualQuery.js";

export default function VisualQueryPage() {
  const { "*": queryString } = useParams();
  const navigate = useNavigate();
  const { isReady, queryOptions } = useContext(DataContext);
  const [input, setInput] = useState(queryString);
  const visualRef = useRef(null);
  const application = useApplicationLoad(isReady);
  const parse = useVisualQuery(isReady, queryString);
  const load = useVisualData(parse.visualQuery);
  const VisualClass = parse.visualQuery?.visualClass;
  const errorMessage = parse.errorMessage || load.errorMessage;
  const isLoading =
    !isReady ||
    !VisualClass ||
    load.datumSet === null ||
    load.loadTimeSeconds === null;
  const steps = useLoadingSteps({
    application,
    datumSet: load.datumSet,
    isLoading,
    isReady,
    load,
    parse,
    VisualClass,
  });
  useEffect(() => setInput(queryString), [queryString]);
  useEffect(() => {
    if (!isLoading && !errorMessage)
      visualRef.current?.scrollIntoView?.({
        behavior: "smooth",
        block: "start",
      });
  }, [errorMessage, isLoading]);
  const loadedQuery =
    load.datumSet?.datumList.length > 0 && load.loadTimeSeconds !== null
      ? queryString
      : null;
  const submit = () => {
    const nextQuery = input.trim();
    if (nextQuery && nextQuery !== queryString) navigate(`/${nextQuery}`);
  };
  return (
    <>
      <QueryMenuAppBar loadedVisualQuery={loadedQuery} />
      <Box className={styles.page}>
        <ChangeViewSection
          value={input}
          onChange={setInput}
          onSubmit={submit}
          queryOptions={queryOptions}
          loadedVisualQuery={loadedQuery}
        />
        {errorMessage ? (
          <Alert severity="error" data-testid="query-error">
            <AlertTitle>Sorry, something went wrong.</AlertTitle>
            {errorMessage}
          </Alert>
        ) : (
          <Box
            data-testid={isLoading ? undefined : "visual-content"}
            ref={visualRef}
          >
            {isLoading ? (
              <LoadingProgress steps={steps} />
            ) : (
              <VisualErrorBoundary key={queryString}>
                <VisualContent
                  VisualClass={VisualClass}
                  datumSet={load.datumSet}
                  loadTimeSeconds={load.loadTimeSeconds}
                  query={parse.visualQuery.query}
                />
              </VisualErrorBoundary>
            )}
          </Box>
        )}
      </Box>
    </>
  );
}
