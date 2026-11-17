# -*- coding: utf-8 -*-
"""v7_cluster_boot2.py — B 组修复版:judge-panel bootstrap(变量阴影 bug 修复)。"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = Path(__file__).resolve().parent.parent / "out" / "x2_v6" / "judge_raw"
SPLIT = 0.60
B = 2000
rng = np.random.default_rng(20261117)

preds = {}
for f in sorted(RAW.glob("*.jsonl")):
    j = f.stem
    for l in f.open(encoding="utf-8"):
        r = json.loads(l)
        preds.setdefault(r["id"], {})[j] = r["winner_pred"]
judges = sorted({j for d in preds.values() for j in d})

ids, flags = [], []
for bid, p in preds.items():          # p = 该 battle 的票 dict
    vals = [v for v in p.values() if v in ("A", "B")]
    if not vals:
        continue
    top = max(set(vals), key=vals.count)
    ids.append(bid)
    flags.append(1 if vals.count(top) / len(vals) <= SPLIT else 0)
n = len(flags)
flags = np.array(flags)
assert abs(flags.mean() - 0.1225) < 5e-5

V = np.array([[1 if p.get(j, "C") == "A" else 2 if p.get(j, "C") == "B" else 0
               for j in judges] for p in (preds[b] for b in ids)])
print("V", V.shape, "| row0 dist:", np.bincount(V[0]))

panel = np.empty(B)
J = len(judges)
for b in range(B):
    Vs = V[:, rng.integers(0, J, J)]
    a = (Vs == 1).sum(1)
    bb = (Vs == 2).sum(1)
    nz = a + bb
    top = np.maximum(a, bb)
    mr = np.where(nz > 0, top / np.maximum(nz, 1), 1.0)
    panel[b] = ((mr <= SPLIT) & (nz > 0)).sum() / n
q90 = float(np.quantile(panel, 0.90))
print(f"judge-panel: mean={panel.mean():.4f} sd={panel.std(ddof=1):.4f} "
      f"q90={q90:.4f} [{panel.min():.4f},{panel.max():.4f}]")

# 合并写入既有 json
OUT = RAW.parent / "v7_cluster_boot.json"
d = json.loads(OUT.read_text(encoding="utf-8"))
d["judge_panel_bootstrap"] = {"B": B, "mean": round(float(panel.mean()), 4),
                              "q90": round(q90, 4),
                              "sd": round(float(panel.std(ddof=1)), 4),
                              "min": round(float(panel.min()), 4),
                              "max": round(float(panel.max()), 4)}
d["fixed"] = "panel-vote-matrix shadowing bug corrected (v2)"
OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("[saved]", OUT)
