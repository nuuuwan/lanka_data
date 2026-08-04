# React JavaScript

## Project structure

```
src/nonview/           Non-UI code
  base/                Generic code, reusable across repos, no repo-specific dependencies
  core/                Repo-specific logic
  constants/           Constants
src/view/              UI code
  atoms/               Single-purpose, stateless components (e.g. Button, Input, Icon)
  moles/               Short for "Molecules": stateless compositions of atoms (e.g. SearchBar = Input + Button)
  organisms/           Stateful components, may compose atoms/moles/other organisms
  pages/               Page-level components, route entry points
```

Within any of these folders, add subfolders by function when it improves organization.

## Where new code goes

Answer in order. The first match wins.

1. Does it touch the DOM or render JSX? If no, it goes in `nonview`.
2. In `nonview`: does it depend on anything specific to this repo (domain models, app constants, API shapes)? If no, it goes in `nonview/base`. If yes, `nonview/core`.
3. Is it a fixed value rather than logic? `nonview/constants`.
4. In `view`: does it hold state (`useState`, `useEffect`, `useReducer`, `useContext`, or any other stateful hook)? If yes, it is an organism, or a page if it is a route entry point.
5. Stateless and composed of other components? A mole.
6. Stateless and composed of nothing but primitives? An atom.

## Rules

### Layering

1. `view` may import from `nonview`. `nonview` must never import from `view`.
2. Atoms and moles must be stateless. All state lives in organisms and pages.
3. Atoms do not import other atoms or moles. Moles compose atoms and other moles.
4. Shared state goes through React Context, defined in `nonview/core` or alongside the organism or page that owns it. Descendants access it through a custom `useXxxContext` hook, never by importing the raw context object.
5. Custom hooks live in `nonview/core`, or `nonview/base` if generic. Name them `useXxx`.

### Files

1. One component per file. The file name matches the component name: `SearchBar.js` exports `SearchBar`.
2. Each component's CSS Module is co-located and named to match: `SearchBar.js` and `SearchBar.module.css`. Import as `import styles from './SearchBar.module.css'` and reference classes via `styles.foo`, never string literals.
3. Target under 100 lines per file. Past that, split into smaller components, mixins, or util classes rather than letting the file grow.

### Values

1. No magic numbers or config values in components. URLs, timeouts, thresholds, and feature flags belong in `nonview/constants`.
2. User-facing copy in JSX (labels, text) may stay inline, unless the file already imports from a copy or i18n source, in which case follow that source.

### Scope of change

1. Do not create unit tests.
2. Delete code that your change leaves unused. Do not delete unrelated code that merely looks unused.
3. When a file you are already editing violates these rules, fix it. Do not refactor untouched files as a side quest; if you notice a violation elsewhere, mention it instead.

## After every code change

Run, in order:

```shell
npx prettier --write --log-level warn src
npx eslint --fix --ext .js src
npx eslint --ext .js src
npm run build
```

All four must pass with no remaining errors. Fix anything that does not.

Then confirm `http://localhost:3000/lanka_data` loads with no errors in the page or the browser console.
