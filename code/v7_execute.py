# -*- coding: utf-8 -*-
"""v7_execute.py — prereg v7 确认运行(judge 生态五步;CPU 分钟级)。
按 prereg_v7.md 冻结判据执行:J1 定律核对/J2 层级复制/J3 census/J4 拆 judge 前瞻。"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as _beta, kendalltau

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = Path(__file__).resolve().parent.parent / "out" / "x2_v6" / "judge_raw"
OUTD = RAW.parent

ALPHA_PORT = 0.20          # battle 争议判据:多数率 ≤ 0.5 + 0.20/2? 冻结口径见 §1.3(b)
SPLIT_CRIT = 0.5 + 0.10    # 多数率 ≤ 0.6 = 争议(预声明:半数分裂带 0.5-0.6)
DELTA = 0.10

judges = sorted(f.stem for f in RAW.glob('*.jsonl'))
B = {}
for j in judges:
    for l in (RAW / f"{j}.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        B.setdefault(r["id"], {})[j] = r["winner_pred"]     # A/B/C(tie)
print(f"judges={len(judges)} battles={len(B)}")

# winner_gt 已冻结;battle 争议 = 17 judge 的预测多数率(排除 tie 后)≤ 阈
contested = {}
major_rate = {}
for bid, prefs in B.items():
    vals = [v for v in prefs.values() if v in ("A", "B")]   # 排除 C=tie 后的方向
    if not vals:
        continue
    top = max(set(vals), key=vals.count)
    mr = vals.count(top) / len(vals)
    major_rate[bid] = mr
    contested[bid] = 1 if mr <= SPLIT_CRIT else 0

n_j = len(contested)
b_share = sum(contested.values()) / n_j
cp_b = 1.0 if b_share >= n_j else float(_beta.ppf(0.9, b_share * n_j + 1, n_j - b_share * n_j))
print(f"[census] battles judged={n_j}; contested(SPLIT_CRIT={SPLIT_CRIT}) 份额={b_share:.4f} "
      f"CP 上界={cp_b:.4f}")

# J2:judge 家族分层(探索性:70B+ vs 7B/8B vs mixtral)
def fam(j):
    if "70B" in j or "72B" in j:
        return "70B-class"
    if "8x7B" in j:
        return "mixtral-8x7B"
    return "7B/8B-class"
fams = {}
for j in judges:
    fams.setdefault(fam(j), []).append(j)
print("families:", {f: len(v) for f, v in fams.items()})

# J1:136 judge 对的定律核对 + τ 矩阵
pref_matrix = {}     # judge -> {bid: direction}
for j in judges:
    pref_matrix[j] = {bid: (1 if v == "A" else 0 if v == "B" else None)
                      for bid, prefs in B.items() if j in prefs
                      for v in [prefs[j]]}
tau_mat, pi_mat = {}, {}
jpairs = []
for a in range(len(judges)):
    for b in range(a + 1, len(judges)):
        ja, jb = judges[a], judges[b]
        common = [bid for bid in B if bid in pref_matrix[ja] and bid in pref_matrix[jb]
                  and pref_matrix[ja][bid] is not None and pref_matrix[jb][bid] is not None]
        if len(common) < 1000:
            continue
        xa = np.array([pref_matrix[ja][bid] for bid in common])
        xb = np.array([pref_matrix[jb][bid] for bid in common])
        flip = float(np.mean(xa != xb))
        tau = kendalltau(xa, xb).statistic
        tau_mat[(ja, jb)] = tau
        pi_mat[(ja, jb)] = flip
        jpairs.append(((ja, jb), flip, tau))
print(f"judge 对: {len(jpairs)};τ 范围 [{min(t for _,_,t in jpairs):.4f},{max(t for _,_,t in jpairs):.4f}]")
dev = max(abs(f - (1 - t) / 2) for _, f, t in jpairs)
print(f"J1 恒等式核对: 最大偏差={dev:.2e} (容差 1e-9)")

# J2 方向:70B-class 内部一致率 vs 7B/8B-class 内部一致率(τ 均值)
def fam_tau(fam):
    js = fams.get(fam, [])
    ts = []
    for a in range(len(js)):
        for b in range(a + 1, len(js)):
            if (js[a], js[b]) in tau_mat:
                ts.append(tau_mat[(js[a], js[b])])
    return float(np.mean(ts)) if ts else None
t70 = fam_tau("70B-class")
t7b = fam_tau("7B/8B-class")
tmix = fam_tau("mixtral-8x7B")
print(f"J2 层级: 70B τ̄={t70} | 7B/8B τ̄={t7b} | mixtral τ̄={tmix}")

# J4:拆 judge 前瞻(偶=校准 8 / 奇=测试 9,按字典序)
srt = sorted(judges)
cal_j, test_j = srt[0::2], srt[1::2]
def fold_stats(fold):
    contested_f = {bid: v for bid, v in contested.items()
                   if bid in B}
    # battle 层:fold 内 judge 的多数率
    out_bad = 0
    for bid in contested_f:
        vals = [pref_matrix[j].get(bid) for j in fold]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        top = max(set(vals), key=vals.count)
        if vals.count(top) / len(vals) <= SPLIT_CRIT:
            out_bad += 1
    return out_bad, len(contested_f)
cal_bad, cal_n = fold_stats(cal_j)
test_bad, test_n = fold_stats(test_j)
print(f"J4: 校准折超率={cal_bad}/{cal_n} | 测试折超率={test_bad}/{test_n}")

out = {"J1_identity_max_dev": float(dev), "J2": {"70B": t70, "7B8B": t7b, "mixtral": tmix},
       "census": {"n": n_j, "contested_share": round(b_share, 4), "cp_bound": round(cp_b, 4)},
       "J4": {"cal_bad": cal_bad, "cal_n": cal_n, "test_bad": test_bad, "test_n": test_n},
       "tau_pairs": {f"{a}|{b}": round(t, 4) for (a, b), t in tau_mat.items()},
       "ts": "2026-11-15 v7"}
(OUTD / "v7_verdict.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
print("[saved] v7_verdict.json")
