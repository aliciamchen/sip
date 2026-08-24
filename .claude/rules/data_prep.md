---
paths:
  - "data_prep/**/*"
---

# Data-prep structure

`data_prep/` contains the raw jsPsych JSON-to-CSV converter and its tests. The converter is the single source of truth for participant exclusions and anonymization; do not reimplement either downstream.

`json_to_csv.py` must fail clearly on unparseable input, missing or duplicate subject IDs, trial and exit-survey mismatches, or zero parsed rows. It normalizes the legacy `neither` intimacy label to `somewhat_formal` while parsing older data.

Run one study with:

```bash
uv run python data_prep/json_to_csv.py <slug>
```

This requires gitignored files in `data/<slug>/raw_data/`. The committed processed CSVs are otherwise sufficient for model and figure work. Run the converter tests directly with `uv run python data_prep/test_json_to_csv.py`, or use `make test` for the complete offline suite.
