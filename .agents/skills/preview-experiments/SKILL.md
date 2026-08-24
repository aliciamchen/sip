---
name: preview-experiments
description: Use when the user asks to open, restart, or check the local experiment/stimuli preview, or reports that an experiment change "doesn't show" or "still looks the same" in the browser.
allowed-tools: Bash, Read, Grep
---

# Serve the experiment preview

The preview page uses ES-module imports and `fetch`, so it must be served over HTTP — `file://` won't work.

- Start: `make preview` (serves `experiments/` at `http://localhost:8000`; run it in the background), then `open http://localhost:8000/preview/`. Individual experiments are at `/<slug>/index.html`.
- Port in use: `lsof -ti tcp:8000 | xargs kill`, then restart. Background servers occasionally die with exit 144 — just restart, it isn't a code problem.
- **"It still looks the same" after an edit is almost always browser cache, not code.** Ask for a hard reload (Cmd+Shift+R) before debugging anything. (A false alarm here once triggered a full puppeteer-probe investigation of working code.)
- After regenerating stimuli (`make experiments`), the server picks up new files, but the browser won't — hard reload again.
- Publishing to the web is a different path: `make deploy-preview` / `bin/deploy-experiment` (which auto-runs the drift check).
