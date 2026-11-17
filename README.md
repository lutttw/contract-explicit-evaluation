# Contract-Explicit Evaluation

Release package for "Contract-Explicit Evaluation: From Auditing Learned Readers to Scoped Certificates" (Applied Intelligence submission).

## Contents

| Directory | Description |
|-----------|-------------|
| `main_merged.tex` + source files | Manuscript (compiles with pdflatex → bibtex → strip_bbl_labels.py → pdflatex ×2) |
| `prereg/` | Preregistrations v1–v8 with amendment ledger |
| `data/` | Frozen verdicts, judge raw records (17 judges × 49,938 battles), stratum certificates, NA-margin flip decomposition |
| `code/` | Analysis scripts (Python; CPU-minutes on released data) |
| `supplementary_sections.tex` | Supplementary material |

## Key artifacts

- `data/v6_verdict.json` — 44-model census verdict (T2, CP 0.146)
- `data/v7_verdict.json` — judge-ecosystem verdict (12.25% contested, CP 0.124)
- `data/v7_e2e6e9.json` — E2 McNemar, E6 validity decomposition, E9 margin-cut sensitivity
- `data/v7_e3_probe.json` — retroactive decode probe (7 resolve / 2 persist)
- `data/v7_e5_kalpha.json` — near-tie stratification + Krippendorff α
- `data/v7_cluster_boot.json` — block + judge-panel bootstrap
- `data/v7_rank_stability.json` — Bradley–Terry rank stability
- `data/v7_stratum_cert.json` — margin-tier certificates

## Reproducibility

All analyses are deterministic on the released frozen artifacts and
require only Python 3 with numpy/scipy. Total compute: CPU-minutes.

## License

CC BY 4.0
