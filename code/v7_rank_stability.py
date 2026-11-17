# -*- coding: utf-8 -*-
"""v7_rank_stability.py — E7:BT/Elo rank stability(2026-11-17;EXPLORATORY)。

R4-F3:battle 级证书 ≠ leaderboard 排序声明。本脚本:
  1) battle id → (model_a, model_b)(battles_meta.json,HF 元数据)
  2) 逐 judge Bradley-Terry(MM 迭代,方向票)+ 多数票 BT → 排序基准
  3) 三口径排序:ALL(全 battle)/ CERTIFIED(margin>0.3)/ HEDGE(margin<=0.3)
  4) 稳定性:certified vs ALL 的 Kendall τ、相邻名次翻转数、top-10 保序
对账:census share==0.1225。→ out/x2_v6/v7_rank_stability.json
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "out" / "x2_v6" / "judge_raw"
META = RAW.parent / "battles_meta.json"
OUT = RAW.parent / "v7_rank_stability.json"
SPLIT = 0.60

meta = json.loads(META.read_text(encoding="utf-8"))
pairs = {r["id"]: (r["model_a"], r["model_b"]) for r in meta}
print(f"meta battles={len(pairs)}")

votes, probs = {}, {}
for f in sorted(RAW.glob("*.jsonl")):
    j = f.stem
    for l in f.open(encoding="utf-8"):
        r = json.loads(l)
        votes.setdefault(r["id"], {})[j] = r["winner_pred"]
        probs.setdefault(r["id"], {})[j] = abs(r["prob_A"] - r["prob_B"])
print(f"judged battles={len(votes)}")

ids = sorted(set(votes) & set(pairs))
print(f"aligned battles={len(ids)}")

# margin 分层(与 v7 同口径)
marg = {b: float(np.mean(list(probs[b].values()))) for b in ids}
tier = {b: ("cert" if marg[b] > 0.3 else "hedge") for b in ids}
flags = {}
for b in ids:
    vals = [v for v in votes[b].values() if v in ("A", "B")]
    if not vals:
        continue
    top = max(set(vals), key=vals.count)
    flags[b] = 1 if vals.count(top) / len(vals) <= SPLIT else 0
share = sum(flags.values()) / len(flags)
assert abs(share - 0.1225) < 5e-5
print(f"census share={share:.4f} ✓")


def bt_fit(battles, iters=800, floor=-25.0):
    """Newman MM(BT)log 空间稳定实现;battles=(winner_model, loser_model)。"""
    models = sorted({m for w, l in battles for m in (w, l)})
    idx = {m: k for k, m in enumerate(models)}
    n = len(models)
    W = np.zeros(n)
    match_cnt = np.zeros((n, n))
    for w, l in battles:
        i, j2 = idx[w], idx[l]
        W[i] += 1
        match_cnt[i, j2] += 1
        match_cnt[j2, i] += 1
    lg = np.zeros(n)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for _ in range(iters):
            diff = lg[:, None] - lg[None, :]
            p = 1.0 / (1.0 + np.exp(-diff))
            denom = (match_cnt * p).sum(1)
            with np.errstate(divide="ignore"):
                newlg = np.where(W > 0, np.log(np.maximum(W, 1e-12)) - np.log(np.maximum(denom, 1e-300)), floor)
            # 阻尼防振荡
            lg = 0.5 * lg + 0.5 * np.clip(newlg, floor, -floor)
    return models, lg


def ranks_of(battles_subset, models_ref=None):
    wins = []
    for b in battles_subset:
        a, bl = pairs[b]
        for j, v in votes[b].items():
            if v == "A":
                wins.append((a, bl))
            elif v == "B":
                wins.append((bl, a))
    models, lg = bt_fit(wins)
    order = np.argsort(-lg)
    rank = {models[i]: k + 1 for k, i in enumerate(order)}
    return rank, models, lg


def ranks_of(battles_subset, models_ref=None):
    wins = []
    for b in battles_subset:
        a, bl = pairs[b]
        for j, v in votes[b].items():
            if v == "A":
                wins.append((a, bl))
            elif v == "B":
                wins.append((bl, a))
    models, strength = bt_fit(wins)
    order = np.argsort(-strength)
    rank = {models[i]: k + 1 for k, i in enumerate(order)}
    return rank, models, strength


battles_all = ids
battles_cert = [b for b in ids if tier[b] == "cert"]
print(f"BT fits: ALL={len(battles_all)} CERT={len(battles_cert)} …")
rank_all, models_all, s_all = ranks_of(battles_all)
rank_cert, models_cert, s_cert = ranks_of(battles_cert)
rank_hedge, _, _ = ranks_of([b for b in ids if tier[b] == "hedge"])

# 稳定性:certified vs ALL
common = [m for m in rank_all if m in rank_cert]
tau = float(np.corrcoef([rank_all[m] for m in common],
                        [rank_cert[m] for m in common])[0, 1])
adj_flip = sum(1 for m in common if rank_all[m] != rank_cert[m])
top10_all = set(sorted(common, key=lambda m: rank_all[m])[:10])
top10_cert = set(sorted(common, key=lambda m: rank_cert[m])[:10])
print(f"certified vs ALL: Kendall ρ={tau:.4f} | 名次变动 {adj_flip}/{len(common)} "
      f"| top-10 重叠 {len(top10_all & top10_cert)}/10")

# 更细稳定性统计
d_rank = {m: abs(rank_all[m] - rank_cert[m]) for m in common}
moved = sorted(d_rank.items(), key=lambda kv: -kv[1])
top20 = set(sorted(common, key=lambda m: rank_all[m])[:20])
top20_max = max(d_rank[m] for m in top20)
top20_overlap = sum(1 for m in top20 if rank_cert[m] <= 20)
ge3 = sum(1 for m, dv in d_rank.items() if dv >= 3)
ge5 = sum(1 for m, dv in d_rank.items() if dv >= 5)
mean_d = float(np.mean(list(d_rank.values())))
out = {
    "battles": {"all": len(battles_all), "cert": len(battles_cert),
                "hedge": len(ids) - len(battles_cert)},
    "cert_vs_all": {"kendall_tau": round(tau, 4), "rank_changes": adj_flip,
                    "n_models": len(common),
                    "top10_overlap": len(top10_all & top10_cert),
                    "top20_overlap": top20_overlap,
                    "top20_max_abs_change": int(top20_max),
                    "moved_ge3": ge3, "moved_ge5": ge5,
                    "mean_abs_change": round(mean_d, 2),
                    "biggest_movers": [(m, dv) for m, dv in moved[:5]]},
    "rank_table_cert": {m: rank_cert[m] for m in sorted(common, key=lambda m: rank_cert[m])[:15]},
    "note": "E7 EXPLORATORY: per-judge direction votes aggregated per battle subset; "
            "Newman MM Bradley-Terry (log-space stable); metadata via hf-mirror "
            "(train.single-turn.json)",
    "ts": "2026-11-17",
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("[saved]", OUT)
