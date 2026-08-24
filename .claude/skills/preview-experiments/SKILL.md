---
name: preview-experiments
description: Use when starting, restarting, opening, or checking the local experiment preview, or when recently changed stimuli appear stale in the browser.
allowed-tools: Bash, Read, Grep
---

# Serve the experiment preview

The preview uses ES modules and `fetch`, so serve it over HTTP:

```bash
make preview
```

Open `http://localhost:8000/preview/`; individual studies are at `http://localhost:8000/<slug>/index.html`.

If port 8000 is occupied, inspect the listener with `lsof -nP -iTCP:8000 -sTCP:LISTEN`. Terminate it only after confirming it is the old preview server, then restart. After code or generated stimuli change, hard-reload the browser before diagnosing stale content.

Publishing is separate: use `make deploy-preview` or `bin/deploy-experiment`, which runs the artifact-drift guard.
