# -*- coding: utf-8 -*-
"""e5_kalpha.py — E5 近平局分层 + W6 Krippendorff α(2026-11-17)。

E5: x2 stratum(44)三维度,配对按 |d1|+|d2|(/2) 分位分层(近平局带 vs 明确带),
    重算 flip rate——检验 gold 维 0.46 的高密度是否近平局噪声驱动(R3-F7)。
Kα: 17 judges × 49,667 battles 的 Krippendorff α(nominal;B=2000 bootstrap CI)。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT_P3 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_P3 / "x2_v4"))

import p3loader as L       # noqa: E402
from p3io import metrics   # noqa: E402

MIRROR_RUNS = ROOT_P3 / "out" / "x2_v4" / "runs"
COMP_RUNS = L.RUNS
DIMS = {"scoring(A_vs_R)": ("score_A", "score_R"),
        "gold(gD_vs_gR)": ("gold_gD", "gold_gR"),
        "threshold(fixed_vs_cal)": ("thr_fixed", "thr_cal")}
CONTRACT_COL = {"score_A": 3, "score_R": 1}

docs = L.canonical_docs(); n = len(docs)
pos = {d: i for i, d in enumerate(docs)}
g = L.GENERATORS["x2"]
sets = L.settings_for("x2")


def build(name, own, runs_dir):
    if runs_dir is not None:
        L.RUNS = runs_dir
    try:
        arrays = L.load_run_arrays("x2", name, ["gD", "gR"])
        own_arr = L.load_run_arrays("x2", name, [own])
    finally:
        L.RUNS = COMP_RUNS
    arrays.update({(f"own@{own}", k[1]): v for k, v in own_arr.items()})
    canon = {k: L.to_canonical(d_, i_, t_, pos, n) for k, (d_, i_, t_) in arrays.items()}
    row = {}
    for c, (sel, var, _c) in sets.items():
        key = (f"own@{own}", var) if sel == "own" else (sel, var)
        a = canon[key]
        m = metrics(a.sum(0, keepdims=True))[0]
        row[c] = float(m[0, CONTRACT_COL.get(c, 2)] - m[1, CONTRACT_COL.get(c, 2)])
    return row


insts = [build(nm, own, None) for nm, own in g["runs"]]
insts += [build(f"ml_{a}_r{r:03d}_s{s}", "gD" if a == "D" else "gR", MIRROR_RUNS)
          for a in ("D", "R") for r in (25, 100)
          for s in ("20261201", "20261202", "20261203", "20261204", "20261205", "20261206")]
print(f"instances={len(insts)}")

out = {}
for dim, (c1, c2) in DIMS.items():
    gaps = [(row.get(c1), row.get(c2)) for row in insts]
    vals = [g for g in gaps if g[0] is not None and g[1] is not None]
    m = len(vals)
    # 配对(i,j)的翻转指示与近平局度量
    flips, mag = [], []
    for i in range(m):
        for j in range(i + 1, m):
            d1 = vals[i][0] - vals[j][0]
            d2 = vals[i][1] - vals[j][1]
            if d1 == 0 or d2 == 0:
                continue
            flips.append(1 if np.sign(d1) != np.sign(d2) else 0)
            mag.append((abs(d1) + abs(d2)) / 2)
    flips = np.array(flips); mag = np.array(mag)
    q = np.quantile(mag, [0.25, 0.5])
    near = flips[mag <= q[0]]
    far = flips[mag >= q[1]]
    res = {"n_pairs": int(len(flips)),
           "pi_all": round(float(flips.mean()), 4),
           "pi_near25": round(float(near.mean()), 4),
           "pi_far75": round(float(far.mean()), 4),
           "near_thr": round(float(q[0]), 3), "far_thr": round(float(q[1]), 3)}
    out[dim] = res
    print(f"{dim}: all={res['pi_all']} near25={res['pi_near25']} far75={res['pi_far75']}")

# ---- Krippendorff α(nominal, 17 judges × 49,667 battles) ----
import json as J
RAW = ROOT_P3 / "out" / "x2_v6" / "judge_raw"
votes, ids = {}, []
for f in sorted(RAW.glob("*.jsonl")):
    j = f.stem
    for l in f.open(encoding="utf-8"):
        r = json.loads(l)
        votes.setdefault(r["id"], {})[j] = r["winner_pred"]
judges = sorted({j for d in votes.values() for j in d})
units = [bid for bid, p in votes.items() if any(v in ("A", "B") for v in p.values())]
units.sort()
U = len(units)
Mx = np.zeros((3, U))          # A/B/C 计数矩阵
cols = {j: k for k, j in enumerate(judges)}
u_idx = {b: k for k, b in enumerate(units)}
for bid in units:
    k = u_idx[bid]
    for j, v in votes[bid].items():
        Mx[{"A": 0, "B": 1, "C": 2, "tie": 2}.get(v, 2), k] += 1

n_judge_votes = Mx.sum()
# 标准 coincidence 形式(nominal):Do=Σ_u (m_u.² − Σ_c m_uc²);De=Σ_{c≠k} n_c n_k
Do = 0.0
for k in range(U):
    s = Mx[:, k].sum()
    if s > 1:
        Do += (s * s - (Mx[0, k] ** 2 + Mx[1, k] ** 2 + Mx[2, k] ** 2)) / (s - 1)
nA, nB, nC = Mx.sum(1)
N = n_judge_votes
De = float(nA * nB + nA * nC + nB * nC) / (N - 1)
alpha = 1.0 - Do / De

(ROOT_P3 / "out" / "x2_v6" / "v7_e5_kalpha.json").write_text(
    json.dumps({"e5_strat": out, "krippendorff_alpha": round(alpha, 4),
                "ts": "2026-11-17 EXPLORATORY"}, ensure_ascii=False, indent=1),
    encoding="utf-8")
print("[saved] v7_e5_kalpha.json")
