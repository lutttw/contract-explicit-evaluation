# -*- coding: utf-8 -*-
"""e2e6e9.py — WP-E 分析批(2026-11-17):E2 McNemar + E6 validity 分解 + E9 margin-cut。

E2: 条款(i)案例(learned 7 ties vs consensus-median 0,155 stable)精确 McNemar;
    J2(learned 7 vs AER 13,44 stable)精确二项(配对表不可得时声明口径)。
E6: judge validity——逐 judge accuracy vs winner_gt;contested 分解为
    human-clear(gt∈{A,B}) vs human-ambiguous(gt=tie);逐层 truth-referenced 表。
E9: margin-cut 0.2/0.3/0.4 的层界敏感性。
对账:census share==0.1225 后产出。→ out/x2_v6/v7_e2e6e9.json
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, beta as _beta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = Path(__file__).resolve().parent.parent / "out" / "x2_v6" / "judge_raw"
OUT = RAW.parent / "v7_e2e6e9.json"
SPLIT = 0.60

# ---- 装载 ----
votes, gts, probs = {}, {}, {}
for f in sorted(RAW.glob("*.jsonl")):
    j = f.stem
    for l in f.open(encoding="utf-8"):
        r = json.loads(l)
        votes.setdefault(r["id"], {})[j] = r["winner_pred"]
        gts[r["id"]] = r["winner_gt"]
        probs.setdefault(r["id"], {})[j] = abs(r["prob_A"] - r["prob_B"])
judges = sorted({j for d in votes.values() for j in d})

ids, flags, marg, gt = [], [], [], []
for bid, p in votes.items():
    vals = [v for v in p.values() if v in ("A", "B")]
    if not vals:
        continue
    top = max(set(vals), key=vals.count)
    mr = vals.count(top) / len(vals)
    ids.append(bid)
    flags.append(1 if mr <= SPLIT else 0)
    marg.append(np.mean(list(probs[bid].values())))
    gt.append(gts[bid])
n = len(ids)
flags = np.array(flags); marg = np.array(marg)
share = flags.mean()
assert abs(share - 0.1225) < 5e-5
print(f"n={n} share={share:.4f}")

# ================= E2 =================
# 条款(i)案例:learned 7 ties vs consensus-median 0(155 stable)——方向确定的双向 McNemar
b_, c_ = 7, 0
p_clause = float(binomtest(int(b_), int(b_) + int(c_), 0.5).pvalue)
print(f"E2 clause(i): learned-ties∧median-not={b_}, 反向={c_}, exact p={p_clause:.4f}")
# J2: learned 7 vs AER 13(44 stable)——总数已知,联合表未存;以"二项近似(主张口径)"
p_j2 = float(binomtest(7, 44, 13 / 44).pvalue)   # learned 7 vs 期望 13/44
print(f"E2 J2: learned 7/44 vs AER-rate 13/44 期望, binomial p={p_j2:.4f}(unpaired 口径)")

# ================= E6 =================
gt_arr = np.array(gt)
clear = gt_arr != "tie"
acc_by_judge = {}
for j in judges:
    a = b = 0
    for i in range(n):
        gtv = gt_arr[i]; pv = votes[ids[i]].get(j, "C")
        if gtv == "tie":
            continue
        a += 1
        if pv == gtv:
            b += 1
    acc_by_judge[j] = round(b / a, 4) if a else None
acc_vals = [v for v in acc_by_judge.values() if v is not None]
print(f"E6 per-judge clear-accuracy: mean={np.mean(acc_vals):.4f} "
      f"[{min(acc_vals):.3f},{max(acc_vals):.3f}]")

contested = flags == 1
decomp = {
    "contested_and_human_clear": int((contested & clear).sum()),
    "contested_and_human_ambig": int((contested & ~clear).sum()),
    "clear_total": int(clear.sum()), "ambig_total": int((~clear).sum()),
}
cs = contested & clear
print(f"E6 contested={contested.sum()}: human-clear {decomp['contested_and_human_clear']} "
      f"({cs.sum()/clear.sum():.4f} of clear) | human-ambiguous {decomp['contested_and_human_ambig']}")

# 逐层 truth-referenced
tiers = [("high(>0.7)", marg > 0.7), ("mid(0.3-0.7)", (marg > 0.3) & (marg <= 0.7)),
         ("low(<=0.3)", marg <= 0.3)]
tier_rows = []
for name, sel in tiers:
    m = sel
    c_ = contested & m
    jc = (c_ & (gt_arr != "tie")).sum()
    ja = (c_ & (gt_arr == "tie")).sum()
    acc = np.mean([acc_by_judge[j] for j in judges])
    tier_rows.append({"tier": name, "n": int(m.sum()),
                      "contested": int(c_.sum()),
                      "contested_human_clear": int(jc),
                      "contested_human_ambig": int(ja),
                      "judge_acc_mean": round(float(acc), 4)})
    print("  ", tier_rows[-1])

# ================= E9 =================
cuts = {}
for cutv in (0.2, 0.3, 0.4):
    sel_hm = marg > cutv
    sel_lo = ~sel_hm
    cp = lambda w, n_: 1.0 if (n_ == 0 or w >= n_) else float(_beta.ppf(0.9, w + 1, n_ - w))
    bh = int(flags[sel_hm].sum()); nh = int(sel_hm.sum())
    bl = int(flags[sel_lo].sum()); nl = int(sel_lo.sum())
    cuts[str(cutv)] = {"cert_n": nh, "cert_share": round(float(flags[sel_hm].mean()), 4),
                       "cert_cp": round(cp(bh, nh), 4),
                       "hedge_n": nl, "hedge_cp": round(cp(bl, nl), 4)}
    print(f"E9 cut={cutv}: cert n={nh} cp={cp(bh,nh):.4f} | hedge n={nl} cp={cp(bl,nl):.4f}")

out = {
    "E2": {"clause_i": {"b": 7, "c": 0, "exact_p": round(p_clause, 4)},
           "j2_binomial_unpaired_p": round(p_j2, 4),
           "note": "J2 joint table not archived; unpaired binomial stated as such"},
    "E6": {"acc_by_judge": acc_by_judge,
           "acc_mean": round(float(np.mean(acc_vals)), 4),
           "acc_min": round(float(min(acc_vals)), 4),
           "acc_max": round(float(max(acc_vals)), 4),
           "decomp": decomp, "tiers": tier_rows},
    "E9": cuts,
    "ts": "2026-11-17 EXPLORATORY",
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("[saved]", OUT)
