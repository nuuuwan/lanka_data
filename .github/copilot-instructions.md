# React JavaScript

## Project structure

```
src/nonview/           Non-UI code
  base/                Generic code, reusable across repos, no repo-specific dependencies
  core/                Repo-specific logic
  constants/           Constants
src/view/              UI code
  atoms/               Single-purpose, stateless components (e.g. Button, Input, Icon)
  moles/           Compositions of atoms, stateless (e.g. SearchBar = Input + Button)
  organisms/           Stateful components, may compose atoms/moles/other organisms
  pages/               Page-level components, route entry points
```

## Rules

- Place new code in the correct directory. If a module in `nonview/core` has no repo-specific dependencies, it belongs in `nonview/base`.
- `view/atoms` and `view/moles` must be stateless: no `useState`, `useEffect`, `useContext`, or other stateful hooks. State lives in `view/organisms` and `view/pages`.
- moles, short for molecules, compose atoms and other moles. Atoms do not compose other atoms or moles.
- No magic numbers or config values (URLs, timeouts, thresholds, feature flags) in components. Put them in `nonview/constants`. User-facing copy (JSX text, labels) may stay inline unless the file already imports from a copy/i18n source.
- UI code may import from `nonview`. `nonview` must never import from `view`.
- One component per file. File name matches the component name (`SearchBar.js` exports `SearchBar`).
- Each component's CSS Module is co-located and named to match: `SearchBar.js` + `SearchBar.module.css`. Import as `import styles from './SearchBar.module.css'` and reference classes via `styles.foo`, never string literals.
- Shared state goes through React Context, defined in `nonview/core` or alongside the `organism`/`page` that owns it. Expose access via a custom hook (`useXxxContext`), not by importing the raw context object into descendants.
- Custom hooks live in `nonview/core` (or `nonview/base` if generic) and are named `useXxx`.
- Do not create unittests

### Large files

- Keep files under 100 lines. If a file grows larger, consider splitting it into smaller, reusable components (atoms or moles) or Mixins or Util classes.

### Organization

- within high level folders (atoms, moles etc) feel free to add sub folders by function for better organization.

## After every code change

Run, in order:

```shell
npx prettier --write --log-level warn src
npx eslint --fix --ext .js src
npx eslint --ext .js src
```

Fix any remaining lint errors before finishing.

Finally, make sure `http://localhost:3000/lanka_data` loads without errors.
