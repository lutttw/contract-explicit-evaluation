# -*- coding: utf-8 -*-
"""x2_stratum44_row.py — 路径 B:v6 census(44)行加入 certmap(2026-11-17)。

装载 24 companion + 20 mirror(x2_v4)run,复用 pc2_certmap 口径
(同 settings_for + CONTRACT_COL + cell_stats),产出三维度 44 行;
双对账:24 行复算 == phase3 C2(逐维精确),44 行 scoring == 0.0803 & bad=3。
任一不符即中止,不产 JSON。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as _beta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT_P3 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_P3 / "x2_v4"))

import p3loader as L       # noqa: E402
from p3io import metrics   # noqa: E402

MIRROR_RUNS = ROOT_P3 / "out" / "x2_v4" / "runs"
OUT = ROOT_P3 / "out" / "x2_v6" / "x2_stratum44_row.json"

CONTRACT_COL = {"score_A": 3, "score_R": 1}          # pc2 口径,其余 col 2
DIMS = {"scoring(A_vs_R)": ("score_A", "score_R"),
        "gold(gD_vs_gR)": ("gold_gD", "gold_gR"),
        "threshold(fixed_vs_cal)": ("thr_fixed", "thr_cal")}
ALPHA_PORT = 0.20
DELTA = 0.10
SEEDS_NEW = ("20261202", "20261203", "20261204", "20261205", "20261206")

docs = L.canonical_docs()                            # 用默认 RUNS(companion)
n = len(docs)
pos = {d: i for i, d in enumerate(docs)}
g = L.GENERATORS["x2"]
sets = L.settings_for("x2")
print(f"canonical docs={n}; x2 companion runs={len(g['runs'])}")


def build_instance(name, own_gold, runs_dir):
    if runs_dir is not None:
        L.RUNS = runs_dir                            # 补丁:镜像 run 装载
    try:
        arrays = L.load_run_arrays("x2", name, ["gD", "gR"])
        own_arr = L.load_run_arrays("x2", name, [own_gold])
    finally:
        L.RUNS = ROOT_P3_COMP_RUNS                   # 恢复
    arrays.update({(f"own@{own_gold}", k[1]): v for k, v in own_arr.items()})
    canon = {k: L.to_canonical(d_, i_, t_, pos, n)
             for k, (d_, i_, t_) in arrays.items()}
    row = {}
    for c, (sel, var, _col) in sets.items():
        key = (f"own@{own_gold}", var) if sel == "own" else (sel, var)
        a = canon[key]
        m = metrics(a.sum(0, keepdims=True))[0]
        row[c] = float(m[0, CONTRACT_COL.get(c, 2)] - m[1, CONTRACT_COL.get(c, 2)])
    p = name.split("_")
    return {"name": name, "arch": "x2", "arm": p[1], "regime": p[2],
            "seed": p[3], "r": row}


ROOT_P3_COMP_RUNS = L.RUNS                           # companion runs 目录

insts20 = [build_instance(nm, own, None) for nm, own in g["runs"]]
g4 = [(f"ml_{arm}_r{rho:03d}_s20261201", "gD" if arm == "D" else "gR")
      for arm in ("D", "R") for rho in (25, 100)]
v6 = [(f"ml_{arm}_r{rho:03d}_s{s}", "gD" if arm == "D" else "gR")
      for arm in ("D", "R") for rho in (25, 100) for s in SEEDS_NEW]
insts_g4 = [build_instance(nm, own, MIRROR_RUNS) for nm, own in g4]
insts_v6 = [build_instance(nm, own, MIRROR_RUNS) for nm, own in v6]
assert len(insts20) == 20 and len(insts_g4) == 4 and len(insts_v6) == 20
print(f"instances: 20 companion + 4 v4-growth + 20 v6 = {len(insts20)+len(insts_g4)+len(insts_v6)}")


def cell_stats(insts, dim):
    c1, c2 = DIMS[dim]
    vals = []
    for inst in insts:
        v1, v2 = inst["r"].get(c1), inst["r"].get(c2)
        if v1 is None or v2 is None:
            return None
        vals.append((v1, v2))
    n_ = len(vals)
    rates = np.zeros(n_)
    flips = pairs = 0
    for i in range(n_):
        fl = tot = 0
        for j in range(n_):
            if i == j:
                continue
            d1 = vals[i][0] - vals[j][0]
            d2 = vals[i][1] - vals[j][1]
            if d1 == 0 or d2 == 0:
                continue
            tot += 1
            pairs += 1
            f = 1 if np.sign(d1) != np.sign(d2) else 0
            fl += f
            flips += f
        rates[i] = fl / tot if tot else 0.0
    bad = int((rates > ALPHA_PORT).sum())
    bound = 1.0 if (n_ == 0 or bad >= n_) else float(_beta.ppf(1 - DELTA, bad + 1, n_ - bad))
    return {"n": n_, "flip_rate": round(flips / pairs, 4) if pairs else None,
            "bad": bad, "bound": round(bound, 4), "rates": rates.tolist()}


row24 = {d: cell_stats(insts20 + insts_g4, d) for d in DIMS}
row44 = {d: cell_stats(insts20 + insts_g4 + insts_v6, d) for d in DIMS}

# ---- 双对账 ----
v4 = json.loads((ROOT_P3 / "out" / "x2_v4" / "phase3_verdict.json").read_text(encoding="utf-8"))
for d in DIMS:
    got, want = row24[d]["flip_rate"], v4["C2"][d]["flip_density"]
    assert abs(got - want) < 5e-4, f"24 行对账失败 {d}: {got} vs {want}"
print("[对账1] 24 行复算 == phase3 C2 逐维精确 ✓",
      {d: row24[d]["flip_rate"] for d in DIMS})
assert abs(row44["scoring(A_vs_R)"]["flip_rate"] - 0.0803) < 5e-4, "44 scoring ≠ 0.0803"
assert row44["scoring(A_vs_R)"]["bad"] == 3, "44 scoring bad ≠ 3"
print("[对账2] 44 行 scoring == 0.0803 & bad=3 ✓")

out = {"row24": {d: {k: v for k, v in row24[d].items() if k != "rates"} for d in DIMS},
       "row44": {d: {k: v for k, v in row44[d].items() if k != "rates"} for d in DIMS},
       "ts": "2026-11-17 path-B"}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("[saved]", OUT)
for d in DIMS:
    print(f"  {d}: 24-row pi={row24[d]['flip_rate']} | 44-row pi={row44[d]['flip_rate']} "
          f"bad={row44[d]['bad']} bound={row44[d]['bound']}")
