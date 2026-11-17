# -*- coding: utf-8 -*-
"""v7_cluster_boot.py — G4/E8:judge census 的聚类稳健性(EXPLORATORY,2026-11-17)。

面板 R2-F2/R4-F2:battles 按 model pair/topic 聚类,i.i.d. CP 可能反保守。
pair 元数据(train.single-turn.json)gated 不可得 → 两个可计算代理:
  A) block bootstrap:按 battle-id 数值序分块(BLK=100),块重抽——若 id 序
     编码了生成顺序(同 pair/topic 相邻),块方差捕获正相关;若随机,结果≈i.i.d.
     (这本身即诊断)。
  B) judge-panel bootstrap:17 judges 有放回重抽,重组 17 票,重算 majority
     contested share(预声明 0.60 判据)——直接测 panel composition 敏感性。
  C) DEFF = 块方差 / 二项方差。
对账:overall share==0.1225 后才产出。产物 out/x2_v6/v7_cluster_boot.json。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = Path(__file__).resolve().parent.parent / "out" / "x2_v6" / "judge_raw"
OUT = RAW.parent / "v7_cluster_boot.json"
SPLIT = 0.60
B = 2000
BLK = 100
rng = np.random.default_rng(20261117)

# ---- 装载(与 v7_execute 同口径) ----
preds = {}
for f in sorted(RAW.glob("*.jsonl")):
    j = f.stem
    for l in f.open(encoding="utf-8"):
        r = json.loads(l)
        preds.setdefault(r["id"], {})[j] = r["winner_pred"]
judges = sorted({j for d in preds.values() for j in d})
print(f"judges={len(judges)} battles={len(preds)}")

ids = []
flags = []          # contested indicator
for bid, prefs in preds.items():
    vals = [v for v in prefs.values() if v in ("A", "B")]
    if not vals:
        continue
    top = max(set(vals), key=vals.count)
    ids.append(bid)
    flags.append(1 if vals.count(top) / len(vals) <= SPLIT else 0)
ids = np.array(ids)
flags = np.array(flags)
n = len(flags)
share = flags.mean()
print(f"census n={n} share={share:.4f}")
assert abs(share - 0.1225) < 5e-5, "对账失败"

# 二项基准方差
var_binom = share * (1 - share) / n

# ---- A) block bootstrap by id order ----
order = np.argsort(ids)
f_sorted = flags[order]
nblk = n // BLK
blocks = f_sorted[: nblk * BLK].reshape(nblk, BLK)
blk_means = blocks.mean(axis=1)
var_block_emp = np.var(blk_means, ddof=1) / nblk   # 块均值方差/块数 ≈ 总体均值方差(块版)
bs_shares = np.empty(B)
for b in range(B):
    pick = rng.integers(0, nblk, nblk)
    bs_shares[b] = blocks[pick].mean()
q90_block = float(np.quantile(bs_shares, 0.90))
deff = var_block_emp / var_binom
print(f"A) block(BLK={BLK}, nblk={nblk}): share 90%分位={q90_block:.4f} "
      f"DEFF={deff:.3f}")

# ---- B) judge-panel bootstrap ----
# 每 battle 的 17 票 → 重抽 judges(有放回)→ 重算 majority contested
votes = {}
for bid in ids:
    votes[bid] = [prefs.get(bid, {}).get(j, "C") for j in judges]
V = np.array([[1 if v == "A" else 2 if v == "B" else 0 for v in votes[bid]] for bid in ids])
panel_shares = np.empty(B)
J = len(judges)
for b in range(B):
    jsel = rng.integers(0, J, J)
    Vs = V[:, jsel]
    a = (Vs == 1).sum(1); bb = (Vs == 2).sum(1)
    nz = (a + bb)
    top = np.maximum(a, bb)
    mr = np.where(nz > 0, top / np.maximum(nz, 1), 1.0)
    panel_shares[b] = ((mr <= SPLIT) & (nz > 0)).sum() / n
q90_panel = float(np.quantile(panel_shares, 0.90))
sd_panel = float(np.std(panel_shares, ddof=1))
print(f"B) judge-panel: share 90%分位={q90_panel:.4f} sd={sd_panel:.4f} "
      f"[{panel_shares.min():.4f},{panel_shares.max():.4f}]")

# ---- 两层证书的 block 版(高+中 margin 层) ----
# margin 需要 probs —— 重算(v7 口径)
probs = {}
for f in sorted(RAW.glob("*.jsonl")):
    j = f.stem
    for l in f.open(encoding="utf-8"):
        r = json.loads(l)
        probs.setdefault(r["id"], {})[j] = abs(r["prob_A"] - r["prob_B"])
marg = np.array([np.mean(list(probs[bid].values())) if probs.get(bid) else np.nan for bid in ids])
ok = ~np.isnan(marg)
tier_bounds = {}
for name, sel in (("high+mid", marg > 0.3), ("low", marg <= 0.3)):
    m = ok & sel
    f_t = flags[m]
    id_t = ids[m]
    o = np.argsort(id_t)
    fs = f_t[o]
    nb = len(fs) // BLK
    if nb < 2:
        continue
    bl = fs[: nb * BLK].reshape(nb, BLK)
    bs = np.array([bl[rng.integers(0, nb, nb)].mean() for _ in range(B)])
    tier_bounds[name] = {"n": int(m.sum()), "share": float(f_t.mean()),
                         "q90_block": float(np.quantile(bs, 0.90))}
    print(f"tier {name}: n={m.sum()} share={f_t.mean():.4f} block-q90={tier_bounds[name]['q90_block']:.4f}")

out = {
    "census": {"n": n, "share": round(float(share), 4), "cp_iid": 0.1244},
    "block_bootstrap": {"blk": BLK, "n_blocks": nblk, "q90": round(q90_block, 4),
                        "deff": round(float(deff), 3)},
    "judge_panel_bootstrap": {"B": B, "q90": round(q90_panel, 4),
                              "sd": round(sd_panel, 4),
                              "min": round(float(panel_shares.min()), 4),
                              "max": round(float(panel_shares.max()), 4)},
    "tiers_block": tier_bounds,
    "note": "pair metadata gated; block-by-id-order is a proxy for pairing/topic "
            "clustering; judge-panel bootstrap measures panel-composition sensitivity",
    "ts": "2026-11-17 G4/E8 EXPLORATORY",
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("[saved]", OUT)
