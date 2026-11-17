# Contract-Explicit Evaluation

Release package for "Contract-Explicit Evaluation: From Auditing Learned Readers to Scoped Certificates" (Applied Intelligence submission).

## Contents

| Directory | Description |
|-----------|-------------|
| `main_merged.tex` + source files | Manuscript (compiles with pdflatex → bibtex → strip_bbl_labels.py → pdflatex ×2) |
| `prereg/` | Preregistrations v1–v8 with amendment ledger |
| `data/` | Frozen verdicts, judge raw records (17 judges × 49,938 battles), stratum certificates, NA-margin flip decomposition |
| `code/` | `verify_release.py` (33-check headline re-derivation on released data) + as-run analysis scripts |
| `supplement_main.pdf` (+ `.tex`/`.bib`) | Online Resource 1: audit-cycle blocks, prereg v2/v3 cycle blocks S1-S10, judge-ecosystem supplement C1-C7, artifact inventory |

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

One command re-derives every headline number from the released artifacts
alone (33 deterministic checks; needs numpy + scipy):

```bash
python code/verify_release.py
```

This re-computes the census (6,083/49,667 = 12.25% contested from the raw
judge records), all Clopper–Pearson bounds, the E2 exact McNemar p, the E3
resolve/persist split, the corrected Krippendorff α, the Bradley–Terry rank
stability, and the J1 concordance identity — see `code/README.md` for the
full check table. As-run producer scripts are included for provenance.

## License

CC BY 4.0
