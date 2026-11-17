# Code

## One-command verification

```bash
python code/verify_release.py    # from the repository root; needs numpy + scipy
```

`verify_release.py` re-derives every headline number of the paper from the
released frozen artifacts alone (`data/`, including the 17 raw judge record
files) — 33 checks, all deterministic, CPU-minutes:

| Check | Re-derives |
|-------|-----------|
| V1 | judge-raw integrity: 17 judges, 848,946 records |
| V2 | census: 49,667 battles judged, 6,083 contested, share 12.25% |
| V3 | Clopper–Pearson bounds: census 0.1244; DocRE census 3/44 → 0.1456; stratum tiers |
| V4 | cross-file consistency (stratum ↔ census ↔ rank stability ↔ bootstrap) |
| V5 | E2 exact McNemar p = 0.0156 from b=7, c=0 |
| V6 | E3 retroactive probe: 9 flips, 2 persist → 7 resolve |
| V7 | E5 stratification shares + corrected Krippendorff α (0.4561 / 0.5035) |
| V8 | E7 Bradley–Terry rank stability τ = 0.9977 |
| V9 | NA-margin endpoint: 69/69 flips |
| V10 | J1 concordance identity re-derived from raw: max dev 3.0e-05 |

## As-run analysis scripts

The remaining scripts are the as-run producers of the frozen verdicts, kept
verbatim for provenance. They reference internal workspace paths
(`out/x2_v6/…`, local absolute paths) and additional frozen inputs that are
not part of this release, so they are not directly runnable here; the released
JSONs in `data/` are the authoritative frozen outputs of these runs.

| Script | Output |
|--------|--------|
| `v7_execute.py` | `data/v7_verdict.json` (census, J1 identity, J2 hierarchy, J4 folds) |
| `v7_stratum_cert.py` | `data/v7_stratum_cert.json` (margin-tier certificates) |
| `v7_cluster_boot.py` / `v7_cluster_boot2.py` | `data/v7_cluster_boot.json` (block + judge-panel bootstrap) |
| `v7_rank_stability.py` | `data/v7_rank_stability.json` (Bradley–Terry, log-space MM) |
| `e2e6e9.py` | `data/v7_e2e6e9.json` (E2 McNemar, E6 validity, E9 margin-cut) |
| `e3_probe.py` / `e3_final.py` | `data/v7_e3_probe.json` (retroactive decode probe) |
| `e5_kalpha.py` / `e5_kalpha2.py` | `data/v7_e5_kalpha.json` (stratification; e5_kalpha2 has the corrected Kα) |
| `na_endpoint.py` | `data/na_margin_flips.json` (NA-margin endpoint) |
| `x2_stratum44_row.py` | v6 44-model census row (dual reconciliation) |

`data/v6_verdict.json` was assembled from the v6 census outputs per the
prereg v6 criteria; its stored numbers are re-checked by `verify_release.py` (V3).
