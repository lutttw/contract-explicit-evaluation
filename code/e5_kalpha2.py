# -*- coding: utf-8 -*-
"""e5_kalpha2.py — Kα 验证:三分类 vs 二分类(仅方向票)+独立抽样对照(2026-11-17)。"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = Path(__file__).resolve().parent.parent / "out" / "x2_v6" / "judge_raw"
votes = {}
for f in sorted(RAW.glob("*.jsonl")):
    j = f.stem
    for l in f.open(encoding="utf-8"):
        r = json.loads(l)
        votes.setdefault(r["id"], {})[j] = r["winner_pred"]
judges = sorted({j for d in votes.values() for j in d})
units = sorted(b for b, p in votes.items() if any(v in ("A", "B") for v in p.values()))
u_idx = {b: k for k, b in enumerate(units)}
J = len(judges)
U = len(units)

def kalpha(mat_counts):
    """Krippendorff nominal α;mat_counts=(K,U) 计数矩阵。"""
    K, U_ = mat_counts.shape
    Do = 0.0
    for k in range(U_):
        s = mat_counts[:, k].sum()
        if s > 1:
            Do += (s * s - (mat_counts[:, k] ** 2).sum()) / (s - 1)
    n = mat_counts.sum(1)
    N = n.sum()
    De = float((n * n).sum() - (n * n).sum())  # placeholder
    De = float(sum(n[c] * n[k] for c in range(K) for k in range(K) if c != k)) / (N - 1)
    return 1.0 - Do / De, Do, De

# 三分类(A/B/C-tie)
Mx = np.zeros((3, U))
for b in units:
    k = u_idx[b]
    for j, v in votes[b].items():
        Mx[{"A": 0, "B": 1}.get(v, 2), k] += 1
a3, Do3, De3 = kalpha(Mx)

# 二分类(剔除 C 票与全 C 单元;unit=有方向票的 battle)
Mx2 = np.zeros((2, U))
used = 0
for b in units:
    k = u_idx[b]
    a = sum(1 for j in judges if votes[b].get(j) == "A")
    bb = sum(1 for j in judges if votes[b].get(j) == "B")
    if a + bb == 0:
        continue
    used += 1
    Mx2[0, k] = a
    Mx2[1, k] = bb
a2, Do2, De2 = kalpha(Mx2[: , :used] if False else Mx2)

# 独立对照:随机配对抽样 pairwise agreement(有向票)
rng = np.random.default_rng(7)
pairs = [(b, j, votes[b][j]) for b in units for j in judges if votes[b].get(j) in ("A", "B")]
arr = np.array([0 if p[2] == "A" else 1 for p in pairs])
idx = rng.integers(0, len(pairs), (200000, 2))
same_unit = pairs[0][0] == pairs[1][0] if False else np.array(
    [pairs[i][0] == pairs[j][0] for i, j in idx[:50000]])
agree = arr[idx[:50000, 0]] == arr[idx[:50000, 1]]
obs_agree = float(agree[same_unit].mean()) if same_unit.any() else float("nan")
print(f"独立抽样:同 battle 票对一致率 = {obs_agree:.4f}(样本 {int(same_unit.sum())})")

print(f"Kα 3-class (A/B/tie): {a3:.4f}  Do={Do3:.0f} De={De3:.0f}")
print(f"Kα 2-class (direction only, {used} units): {a2:.4f}  Do={Do2:.0f} De={De2:.0f}")

out = {"alpha_3class": round(float(a3), 4), "alpha_2class_direction": round(float(a2), 4),
       "pairwise_agreement_samebattle_sampled": round(obs_agree, 4),
       "chance_3class": round(float((Mx.sum(1) / Mx.sum()) ** 2 @ np.ones(3)), 4),
       "ts": "2026-11-17 verification"}
(Path(__file__).resolve().parent.parent / "out" / "x2_v6" / "v7_kalpha_verify.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("[saved] v7_kalpha_verify.json")
