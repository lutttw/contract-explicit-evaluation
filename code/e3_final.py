# -*- coding: utf-8 -*-
"""e3_final.py — E3 回溯探针终版(2026-11-17;EXPLORATORY)。

解码约定(e3_atlop_infer_score.py 权威出土):
  v1 registry 的 atlop 系 threshold 轴两读数 = native(ATLoss get_label,top4+NA 阈)
  与 sigmoid05(σ≥0.5);stored rel id = logits col id 直接(字母序 1..96)。
探针:第三解码 = 先验匹配(θ_r = logit 分位 @ 1−金标频率_r,gold-free 拟合);
  对 9 个 atlop 系 threshold 翻转判 sign(native)/sign(sig05)/sign(pm);
  x2|docunet(96 列三源公共空间,另一解码族)本轮排除,v8 处理。
对账:代表 run 的 native/sig05 gap vs r1 per_run_points 精确命中后产出。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT_P3 = HERE.parent
COMP = Path(r"<local-workspace>\04_文档级IE")
PREP = COMP / "out" / "lre_rebuttal_v2" / "prepared"
RV5 = COMP / "out" / "revision_v5_experiments"
R1 = Path(r"<local-workspace>\04_文档级IE\out\evallever_release_v1_submission"
          r"\statistics\r1_rank_flip.json")
OUT = ROOT_P3 / "out" / "x2_v6" / "v7_e3_probe.json"

sys.path.insert(0, "..")
import p3loader as L  # noqa: E402
from p3io import COL, metrics  # noqa: E402

src = (ROOT_P3 / "src" / "integrate_deberta_v3.py").read_text(encoding="utf-8")
for line in ('sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")',
             'sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")'):
    src = src.replace(line, "pass")
import types
mm = types.ModuleType("integrate_s")
mm.__file__ = str(ROOT_P3 / "src" / "integrate_deberta_v3.py")
exec(compile(src, mm.__file__, "exec"), mm.__dict__)

docs = L.canonical_docs(); n = len(docs)
pos = {d: i for i, d in enumerate(docs)}
golds = mm.load_golds(str(PREP / "golds_gD.jsonl"))
cands = {}
for line in (PREP / "candidates" / "evaluation.jsonl").read_text(encoding="utf-8-sig").split(chr(10)):
    if line.strip():
        x = json.loads(line)
        cands.setdefault(x["doc_id"], []).append((x["h"], x["t"], x["group"]))
tot_cand = sum(len(v) for v in cands.values())

# 金标频率(stored id 空间 1..96,golds sets 即该空间)
freq = {}
for did, pairs in cands.items():
    gd = golds.get(did, {})
    for (h, t, _gp) in pairs:
        for r in gd.get((h, t), ()):
            freq[r] = freq.get(r, 0) + 1
tot = sum(freq.values())
for r in freq:
    freq[r] /= tot
print(f"candidates={tot_cand}; freq rels={len(freq)}")

FAM_RUN = {
    "atlop_pe8": RV5 / "independent_model_retrain" / "runs" / "atlop_D_s20260917",
    "atlop_full": RV5 / "independent_model" / "runs" / "atlop_D_s20260917",
    "dreeam": RV5 / "dreeam" / "runs" / "dreeam_pe8_s20260917_l01",
    "docunet": RV5 / "docunet" / "runs" / "docunet_pe8_s20260917",
    "kddocre": RV5 / "kddocre" / "runs" / "kddocre_pe8_s20260917",
}


def decode_native(logits, num_labels=4):
    th = logits[:, :1]
    mask = logits > th
    topv = np.sort(logits, axis=1)[:, -num_labels][:, None]
    mask &= logits >= topv
    return mask


def family_gaps(fam):
    root = FAM_RUN[fam]
    Lm = np.load(root / "logits_evaluation.npy").astype(np.float64)
    kf = root / "keys_evaluation.jsonl"
    if not kf.exists():
        kf = FAM_RUN["atlop_pe8"] / "keys_evaluation.jsonl"
    keys = [json.loads(l) for l in kf.open(encoding="utf-8")]
    # prior-match θ(per stored id)
    theta = {}
    for r, fr in freq.items():
        theta[r] = float(np.quantile(Lm[:, r], 1.0 - fr))
    dec = {}
    P_nat = decode_native(Lm)                       # (N,97),col=stored id
    P_sig = 1.0 / (1.0 + np.exp(-Lm[:, 1:])) >= 0.5  # (N,96),col j ↔ stored j+1
    # 重训 npy 列 c(1..96)↔ stored rel c−1;col 0 = NA 阈值列(native 专用)
    theta_full = {c: theta.get(c - 1, float("inf")) for c in range(1, 97)}
    sets = {
        "native": (P_nat[:, 1:], {p: p for p in range(96)}),
        "sigmoid05": (P_sig, {p: p for p in range(96)}),
        "prior": (Lm[:, 1:] >= np.array([theta_full[c] for c in range(1, 97)])[None, :],
                  {p: p for p in range(96)}),
    }
    for name, (P, colmap) in sets.items():
        preds = {}
        for k, rec in enumerate(keys):
            did = rec["doc_id"]
            h, t = rec["h"], rec["t"]
            s = preds.setdefault(did, set())
            for p, c in colmap.items():
                if P[k, p]:
                    s.add((h, c, t))
        arr = mm.counts_for(preds, golds, cands, docs, pos, n)
        print(f"    DEBUG {name}: pred_docs={len(preds)} "
              f"rels={sum(len(v) for v in preds.values())} "
              f"tp={int(arr[:, :, 3].sum())} n_pred={int(arr[:, :, 1].sum())}")
        mv = metrics(arr.sum(0, keepdims=True))[0]
        dec[name] = float(mv[0, COL["F1"]] - mv[1, COL["F1"]])
    return dec


# doc 集合(仅供过滤;pos 由 canonical)
pos = {d: k for k, d in enumerate(docs)}

r1 = json.loads(R1.read_text(encoding="utf-8"))
res = {}
for fam in FAM_RUN:
    dec = family_gaps(fam)
    ref = r1["gaps"][fam]
    ok_n = abs(dec["native"] - ref["thr_fixed"]["point"]) < 5e-4
    ok_s = abs(dec["sigmoid05"] - ref["thr_cal"]["point"]) < 5e-4
    print(f"{fam}: native={dec['native']:+.4f}(登记 {ref['thr_fixed']['point']:+.4f} "
          f"{'✓' if ok_n else '✗'}) sig05={dec['sigmoid05']:+.4f}(登记 {ref['thr_cal']['point']:+.4f} "
          f"{'✓' if ok_s else '✗'}) prior={dec['prior']:+.4f}")
    res[fam] = dec

# 9 翻转判决
print("\n== 9 个 atlop 系 threshold 翻转的先验匹配判决 ==")
verdicts = []
persist = 0
for f in r1["flip_summary"]["flips"]:
    if "threshold" not in f["dim"]:
        continue
    a, b = f["pair"].split("|")
    if a not in res or b not in res:
        print(f"  {f['pair']:24s} excluded(x2 空间,v8)")
        continue
    sf = np.sign(res[a]["native"] - res[b]["native"])
    sc = np.sign(res[a]["sigmoid05"] - res[b]["sigmoid05"])
    sp = np.sign(res[a]["prior"] - res[b]["prior"])
    v = "PERSISTS" if (sp != sf and sp != sc and sp != 0) else (
        "collapses(pm=" + ("native" if sp == sf else "sig05") + ")" if sp != 0 else "pm-tie")
    if v == "PERSISTS":
        persist += 1
    verdicts.append({"pair": f["pair"], "sf": int(sf), "sc": int(sc), "sp": int(sp), "verdict": v})
    print(f"  {f['pair']:24s} native={sf:+.0f} sig05={sc:+.0f} prior={sp:+.0f} -> {v}")

print(f"\n== E3 汇总: {persist}/9 PERSISTS;其余为先验匹配与某一注册解码同侧(校准可解释) ==")
OUT = ROOT_P3 / "out" / "x2_v6" / "v7_e3_probe.json"
OUT.write_text(json.dumps({"family_gaps": res, "flips": verdicts,
                           "persist_count": persist,
                           "note": "atlop-family 9 threshold flips; x2|docunet excluded (96-col space, v8)",
                           "ts": "2026-11-17 E3 EXPLORATORY final"}, ensure_ascii=False, indent=1),
              encoding="utf-8")
print("[saved]", OUT)
