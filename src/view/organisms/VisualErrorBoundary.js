import { Alert, AlertTitle } from "@mui/material";
import { Component } from "react";

export default class VisualErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error(
      "[VisualErrorBoundary] Visualization failed",
      error,
      errorInfo,
    );
  }

  render() {
    if (this.state.hasError) {
      return (
        <Alert severity="error" data-testid="query-error">
          <AlertTitle>
            Sorry, we couldn&apos;t show this visualization.
          </AlertTitle>
          Please check your request and try again.
        </Alert>
      );
    }

    return this.props.children;
  }
}
