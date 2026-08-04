import { Tab, Tabs } from "@mui/material";

export default function QueryModeTabs({ mode, onChange }) {
  return (
    <Tabs
      value={mode}
      onChange={(_event, nextMode) => onChange(nextMode)}
      aria-label="Query input mode"
      sx={{
        borderBottom: 1,
        borderColor: "divider",
        mb: 2,
        minHeight: 40,
        "& .MuiTab-root": {
          minHeight: 40,
          px: 2.5,
          textTransform: "none",
        },
        "& .Mui-selected": {
          color: "text.primary",
          fontWeight: 700,
        },
      }}
    >
      <Tab label="Expert" value="expert" />
      <Tab label="Layperson" value="layperson" />
    </Tabs>
  );
}
