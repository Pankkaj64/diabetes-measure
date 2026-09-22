# Frontend — DiaPredict AI Dashboard

React 19 + TypeScript + Vite 8 single-page dashboard. No UI library, no state library —
every component, chart, gauge and icon is hand-built with inline SVG and CSS.

See the [root README](../README.md) for architecture, the API contract and the ML pipeline.

## Requirements

**Node 20.19+ or 22.12+.** Vite 8 will not build on Node 18.

## Run

```bash
cd frontend && npm install
```

```bash
cd frontend && npm run dev
```

Opens on http://localhost:5173 and proxies `/api` to the backend.

## Scripts

| Script | Does |
| --- | --- |
| `npm run dev` | Dev server with HMR |
| `npm run build` | `tsc -b` then production build to `dist/` |
| `npm run lint` | oxlint |
| `npm run preview` | Serve the built `dist/` |

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_TARGET` | `http://localhost:8000` | Backend origin for the `/api` proxy |

```bash
VITE_API_TARGET=http://localhost:8001 npm run dev
```

## Structure

```
src/
├── App.tsx      Everything: types, nav config, icons, views, derived metrics
├── index.css    Design tokens, components, responsive rules
└── main.tsx     Entry point
```

### `App.tsx` map

| Region | Contents |
| --- | --- |
| Types | API response interfaces |
| `NAV_GROUPS` | Sidebar structure — add a tab by adding one entry |
| `CLINICAL_PRESETS` | Four reference patient profiles with risk tiers |
| `Icon` | Inline SVG set, colour bound to `currentColor` |
| `clinicalMetrics()` | Sensitivity, specificity, PPV, NPV, balanced accuracy, MCC, Youden's J from a confusion matrix |
| `EmptyState` | Shared empty/offline placeholder |
| `App` | State, `loadAllData()`, `runPrediction()`, and the six views |

## How data flows

`loadAllData()` runs once on mount. It checks `/api/health` first, then fetches the
four data endpoints **in parallel with independent fallbacks** — a single failing
endpoint degrades one view instead of blanking the dashboard. It finishes by scoring
patient #1 so the calculator is never empty on arrival.

Every view derives from that same fetched state, so figures cannot disagree between
tabs. The Evaluation view computes all of its clinical metrics locally from the
confusion matrices in `/api/models` — no extra request.

When the backend is unreachable the app still renders: an offline banner, a status
pill, empty states per view, and a **Retry Connection** button that re-runs the loader
in place without a page reload.

## Styling

Design tokens are declared on `:root` at the top of `index.css`.

- Neutral slate surfaces, one accent blue; semantic red/green/amber reserved for
  meaning (risk, outcome, status) rather than decoration
- Tabular numerals on metrics so columns align
- Breakpoints at 1024 / 768 / 480 px; the sidebar becomes an off-canvas drawer and
  wide tables scroll internally instead of forcing horizontal page scroll
- Honours `prefers-reduced-motion`, and disables sticky hover styles under `hover: none`

## Adding a view

1. Add an entry to `NAV_GROUPS` (its `id` widens the `TabId` union automatically).
2. Add an icon case to `Icon` if you need a new one.
3. Render `{activeTab === 'your-id' && ( … )}` in `<main>`.
