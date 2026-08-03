import { render } from "@testing-library/react";
import { useState } from "react";

import VisualQueryForm from "./VisualQueryForm.js";

export const VISUAL_QUERY = "Person/Time=2024+Province+Religion/Count/BarChart";
export const QUERY_OPTIONS = {
  entities: ["House", "Person", "Vote"],
  dimensionsByEntity: {
    Person: ["Time", "Province", "Religion", "District"],
  },
};

export function setClipboard(clipboard) {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: clipboard,
  });
}

export function StatefulVisualQueryForm({ onChange = jest.fn() }) {
  const [value, setValue] = useState(VISUAL_QUERY);
  return (
    <VisualQueryForm
      value={value}
      onChange={(nextValue) => {
        setValue(nextValue);
        onChange(nextValue);
      }}
      onSubmit={jest.fn()}
      queryOptions={QUERY_OPTIONS}
    />
  );
}

export function renderForm(props = {}) {
  return render(
    <VisualQueryForm
      value={VISUAL_QUERY}
      onChange={jest.fn()}
      onSubmit={jest.fn()}
      queryOptions={QUERY_OPTIONS}
      {...props}
    />,
  );
}
