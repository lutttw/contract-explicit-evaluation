# -*- coding: utf-8 -*-
"""v7_stratum_cert.py — 分层条件证书(2026-11-16;EXPLORATORY post-hoc,v7 冻结后)。

对 fig_judge_data 同口径的 margin 分层,逐层算 census CP 上界(δ=0.10):
  cp(b,n) = Beta.ppf(0.9, b+1, n-b);b=0 时 = 1-0.9^(1/n)
产出到 out/x2_v6/v7_stratum_cert.json;数字与稿内冻结分层对账后才写。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as _beta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = Path(__file__).resolve().parent.parent / "out" / "x2_v6" / "judge_raw"

probs, preds = {}, {}
for f in RAW.glob("*.jsonl"):
    j = f.stem
    for l in f.open(encoding="utf-8"):
        r = json.loads(l)
        probs.setdefault(r["id"], {})[j] = (r["prob_A"], r["prob_B"], r["prob_C"])
        preds.setdefault(r["id"], {})[j] = r["winner_pred"]

SPLIT = 0.60
contested = {}
for bid, prefs in preds.items():
    vals = [v for v in prefs.values() if v in ("A", "B")]
    if not vals:
        continue
    top = max(set(vals), key=vals.count)
    contested[bid] = 1 if (vals.count(top) / len(vals)) <= SPLIT else 0

def margin_of(bid):
    return float(np.mean([abs(pA - pB) for (pA, pB, _pC) in probs[bid].values()]))

cp = lambda b, n: 1.0 if (n == 0 or b >= n) else float(_beta.ppf(0.9, b + 1, n - b))

strata = [("high", lambda m: m > 0.7), ("mid", lambda m: 0.3 < m <= 0.7), ("low", lambda m: m <= 0.3)]
out = {}
for name, sel in strata:
    ids = [b for b in contested if sel(margin_of(b))]
    n = len(ids)
    b = int(sum(contested[b_2] for b_2 in ids))
    out[name] = {"n": n, "bad": b, "share": round(b / n, 4) if n else None,
                 "cp_bound": round(cp(b, n), 6)}
    print(f"{name}: n={n:,} bad={b} share={b/n:.4f} CP={cp(b, n):.6f}")

n_all = sum(v["n"] for v in out.values())
b_all = sum(v["bad"] for v in out.values())
out["overall"] = {"n": n_all, "bad": b_all, "share": round(b_all / n_all, 4),
                  "cp_bound": round(cp(b_all, n_all), 6)}
print(f"overall: n={n_all:,} bad={b_all} share={b_all/n_all:.4f} CP={cp(b_all, n_all):.4f}")
# 对账:census 头条(high 层 bad=1 与稿内 0.0% 舍入一致;P=1/5994)
assert n_all == 49667 and abs(out["overall"]["share"] - 0.1225) < 5e-5 and abs(out["overall"]["cp_bound"] - 0.1244) < 5e-4, "census 对账失败"
assert out["high"]["bad"] <= 1 and out["high"]["share"] < 5e-4, f"high 层异常: {out['high']}"

Path(__file__).resolve().parent.parent.joinpath("out/x2_v6/v7_stratum_cert.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("[saved] v7_stratum_cert.json — RECONCILED")
