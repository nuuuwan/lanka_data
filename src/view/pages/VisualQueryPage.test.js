import { fireEvent, render, screen } from "@testing-library/react";
import { useNavigate, useParams } from "react-router-dom";
import DataContext from "../../nonview/core/data_context/DataContext.js";
import VisualQueryPage from "./VisualQueryPage.js";

jest.mock("react-router-dom", () => ({
  ...jest.requireActual("react-router-dom"),
  useNavigate: jest.fn(),
  useParams: jest.fn(),
}));

test("updates the URL when the visual query is submitted", () => {
  const navigate = jest.fn();
  useNavigate.mockReturnValue(navigate);
  useParams.mockReturnValue({
    "*": "Person/Time=2024+Province+Religion/Count/Blocks",
  });

  render(
    <DataContext.Provider value={{ isReady: false }}>
      <VisualQueryPage />
    </DataContext.Provider>,
  );

  const input = screen.getByRole("textbox", { name: "Visual query" });
  fireEvent.change(input, {
    target: {
      value: " Person/Time=2024+District+Religion/Count/BarChart ",
    },
  });
  fireEvent.submit(screen.getByRole("form", { name: "Visual query form" }));

  expect(navigate).toHaveBeenCalledWith(
    "/Person/Time=2024+District+Religion/Count/BarChart",
  );
});
