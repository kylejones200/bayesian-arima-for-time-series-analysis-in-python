# Data provenance

## `north_dakota_oil_monthly.csv`

- **Source:** Aggregated from the North Dakota industrial commission / PPDM-style well-level file (`north_dakota_production.csv`).
- **Coverage:** Monthly state-wide sum of reported **Oil** (barrels), Jan 2016–Nov 2024 (107 months).
- **Regenerate:**

```bash
uv run python scripts/rebuild_monthly.py /path/to/north_dakota_production.csv --metric oil
```

Do not commit local paths to the full well-level file; use `config.local.yaml` (see `config.local.yaml.example`).
