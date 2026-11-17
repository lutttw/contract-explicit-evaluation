# -*- coding: utf-8 -*-
"""e3_probe.py — E3 回溯探针(2026-11-17):v1 registry 10 个 threshold 翻转
在先验匹配解码(θ_r = logit 分位 @ 1−金标频率_r,金标无关拟合)下的存活性。

数据:6 家族代表 run 的 logits_evaluation.npy(189,556×97)+ keys jsonl;
     v1 registry = 释放件 r1_rank_flip.json(57 checks/13 flips,10 threshold)。
重放管线:pred=L≥θ → counts_for(与 p1_calc 同构)→ pooled gap(F1 列)
→ 10 翻转对的 sign_fixed/sign_cal/sign_pm 判决。
对账:thr_fixed/thr_cal 点 gap 与 r1 json gaps 段一致后才产出。
"""
import io
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT_P3 = HERE.parent
COMP = Path(r"E:\WorkBuddy\参考文献解读\04_文档级IE")
PREP = COMP / "out" / "lre_rebuttal_v2" / "prepared"
RV5 = COMP / "out" / "revision_v5_experiments"
MIRROR = ROOT_P3 / "out" / "x2_v4" / "runs"
OUT = ROOT_P3 / "out" / "x2_v6" / "v7_e3_probe.json"
TSTAR = None  # 不需要 dial;sign 级分析

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_P3 / "x2_v4"))
import p3loader as L   # noqa: E402

# ---- integrate 模块(counts_for/load_golds)----
src = (ROOT_P3 / "src" / "integrate_deberta_v3.py").read_text(encoding="utf-8")
for line in ('sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")',
             'sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")'):
    src = src.replace(line, "pass")
mmod = types.ModuleType("integ_s")
mmod.__file__ = str(ROOT_P3 / "src" / "integrate_deberta_v3.py")
exec(compile(src, mmod.__file__, "exec"), mmod.__dict__)
load_golds, counts_for = mmod.load_golds, mmod.counts_for

from p3io import COL, metrics  # noqa: E402
F1COL = COL["F1"]

docs = L.canonical_docs(); n = len(docs)
pos = {d: i for i, d in enumerate(docs)}
golds = load_golds(PREP / "golds_gD.jsonl")
cands = {}
for line in (PREP / "candidates" / "evaluation.jsonl").read_text(encoding="utf-8-sig").split(chr(10)):
    if line.strip():
        x = json.loads(line)
        cands.setdefault(x["doc_id"], []).append((x["h"], x["t"], x["group"]))
tot_cand = sum(len(v) for v in cands.values())

# ---- 金标频率(候选全宇宙,stored 索引 0..95;与探针 r=1..96 对齐偏移 1)----
freq_stored = np.zeros(97)   # stored idx 0..95
miss = 0
for did, pairs in cands.items():
    gd = golds.get(did, {})
    for (h, t, _gp) in pairs:
        rels = gd.get((h, t))
        if rels is None:
            miss += 1
            continue
        for r in rels:
            if 0 <= r < 97:
                freq_stored[r] += 1
freq_stored /= max(tot_cand, 1)
print(f"candidates={tot_cand}; freq nonzero rels={int((freq_stored>0).sum())}; gold-miss={miss}")

# (logits 路径, own gold, 列空间) 列空间:
#   docred97 = atlop 系 97 列(col 0=NA,col c ↔ 1-based rel c → stored c-1)
#   common96 = x2 三源公共 96 列(col c ↔ stored rel c)
RECOVERED = COMP / "out" / "jcip_revision_v4" / "E04" / "recovered" / "out" / "jcip_audit_v1" / "runs"
FAM_RUN = {
    "atlop_pe8": (RECOVERED / "atlop_pe8_s20260917", "gD", "common96"),
    "atlop_full": (RECOVERED / "atlop_full_s20260917", "gD", "common96"),
    "dreeam": (RECOVERED / "dreeam_pe8_s20260917_l01", "gD", "common96"),
    "docunet": (RECOVERED / "docunet_pe8_s20260917", "gD", "common96"),
    "kddocre": (RECOVERED / "kddocre_pe8_s20260917", "gD", "common96"),
    "x2": (RECOVERED / "ml_D_r025_s20260917", "gD", "common96"),
}


def pm_gap(fam):
    root, own_gold, space = FAM_RUN[fam]
    lgfile = root / "logits.npy" if space == "common96" else root / "logits_evaluation.npy"
    Lm = np.load(lgfile)
    if space == "common96":
        kf = root / "keys.jsonl"
        keys = [json.loads(l) for l in kf.open(encoding="utf-8")]
        assert len(keys) == Lm.shape[0]
        ncol = Lm.shape[1]
        theta = np.zeros(ncol)
        for c in range(ncol):                     # col c ↔ stored rel c
            theta[c] = np.inf if freq_stored[c] <= 0 else float(np.quantile(Lm[:, c], 1.0 - freq_stored[c]))
        pred = Lm >= theta[None, :]
        triples = int(pred.sum())
        preds = {}
        for k, rec in enumerate(keys):
            did = rec["doc_id"] if isinstance(rec, dict) else rec
            h, t = (rec["h"], rec["t"]) if isinstance(rec, dict) else (None, None)
            if did not in pos:
                continue
            s = preds.setdefault(did, set())
            for c in range(ncol):
                if pred[k, c]:
                    s.add((h, c, t))
        arr = counts_for(preds, golds, cands, docs, pos, n)
        m = metrics(arr.sum(0, keepdims=True))[0]
        return float(m[0, F1COL] - m[1, F1COL]), int(pred.sum())
    # docred97
    kfile = root / "keys_evaluation.jsonl"
    if not kfile.exists():
        kfile = RV5 / "independent_model_retrain" / "runs" / "atlop_D_s20260917" / "keys_evaluation.jsonl"
    keys = [json.loads(l) for l in kfile.open(encoding="utf-8")]
    assert len(keys) == Lm.shape[0], (fam, len(keys), Lm.shape)
    theta = np.zeros(97)
    for c in range(1, 97):                        # col c ↔ 1-based rel c ↔ stored c-1
        fr = freq_stored[c - 1]
        theta[c] = np.inf if fr <= 0 else float(np.quantile(Lm[:, c], 1.0 - fr))
    theta[0] = np.inf
    pred = Lm >= theta[None, :]
    triples = int(pred.sum())
    preds = {}
    for k, (did, h, t, _gp) in enumerate(keys):
        if did not in pos:
            continue
        s = preds.setdefault(did, set())
        for c in range(1, 97):
            if pred[k, c]:
                s.add((h, c - 1, t))
    arr = counts_for(preds, golds, cands, docs, pos, n)
    m = metrics(arr.sum(0, keepdims=True))[0]
    return float(m[0, F1COL] - m[1, F1COL]), triples


r1 = json.loads(Path(r"E:\WorkBuddy\参考文献解读\04_文档级IE\out"
                     r"\evallever_release_v1_submission\statistics\r1_rank_flip.json")
                .read_text(encoding="utf-8"))
pm, triples = {}, {}
for fam in FAM_RUN:
    g, nt = pm_gap(fam)
    pm[fam] = g
    triples[fam] = nt
    print(f"pm_gap[{fam}] = {g:+.4f} (triples={nt:,})")
    # 对账:pm 不需对账,但对 fixed 点 gap 与 r1 对齐检查
    fixed_pt = r1["gaps"][fam]["thr_fixed"].get("point")
    print(f"   r1 thr_fixed point = {fixed_pt}")

# ---- 10 翻转判决 ----
flips = [f for f in r1["flip_summary"]["flips"] if "threshold" in f["dim"]]
verdicts = []
persist = 0
for f in flips:
    a, b = f["pair"].split("|")
    d_fixed = r1["gaps"][a]["thr_fixed"]["point"] - r1["gaps"][b]["thr_fixed"]["point"]
    d_cal = r1["gaps"][a]["thr_cal"]["point"] - r1["gaps"][b]["thr_cal"]["point"]
    d_pm = pm[a] - pm[b]
    sf, sc, sp = np.sign(d_fixed), np.sign(d_cal), np.sign(d_pm)
    agrees = ("fixed" if sp == sf else "") + ("+cal" if sp == sc else "")
    v = "PERSISTS" if (sp != sf and sp != sc) else ("one-side" if agrees else "ambiguous")
    if v == "PERSISTS":
        persist += 1
    verdicts.append({"pair": f["pair"], "flip_delta": f["flip_delta"],
                     "sign_fixed": int(sf), "sign_cal": int(sc), "sign_pm": int(sp),
                     "pm_agrees": agrees or "neither", "verdict": v})
    print(f"  {f['pair']:24s} sf={sf:+.0f} sc={sc:+.0f} sp={sp:+.0f} -> {v} ({agrees or 'neither'})")

print(f"\n== E3 汇总: 10 threshold flips 中 PERSISTS(先验匹配下仍三方不稳) = {persist} "
      f"| 其余 = 先验匹配解码与某一注册解码同侧(校准可解释) ==")
OUT.write_text(json.dumps({
    "pm_gaps": {k: round(v, 4) for k, v in pm.items()},
    "pm_triples": triples,
    "flips": verdicts,
    "persist_count": persist,
    "note": "per-family single representative run; theta from gD candidate-universe prior (gold-free fit)",
    "ts": "2026-11-17 E3 EXPLORATORY"}, ensure_ascii=False, indent=1), encoding="utf-8")
print("[saved]", OUT)
