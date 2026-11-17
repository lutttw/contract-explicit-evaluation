# -*- coding: utf-8 -*-
"""na_endpoint.py — NA 对 spurious-density 契约端点(§7 自名社区争议之三,NA-handling)。

定义:NA 对 = 候选对中无金标关系的对;spurious-pair density =
  fp_pair(预测在无金标对上的对数)/ NA 对数(n_cand - n_gold_pairs)
读数 = density_intra - density_inter(跨句 NA 幻觉差)
新轴:gold(NA_margin)(microF1_gap vs NA_spurious_gap 的配对翻转——
  即同一对生成器在'质量读数'与'NA 幻觉读数'上的排序是否一致)

范围:全部 16 计数可得族(7 冻结 per_doc + 8 新族三元组重算 + deberta/longformer);
LLM 族因 M2 排除不入。boot 同冻结口径。
"""
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import types as _types

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p3loader as L  # noqa: E402
from p3io import bootweights  # noqa: E402

ROOT = Path(r"<local-workspace>\paper3_stable_agg")
PREP = Path(r"<local-workspace>\04_文档级IE\out\lre_rebuttal_v2\prepared")

# NA spurious density 逐文档(boot 需要文档级):fp_pair / max(1, n_cand - n_gold_pairs)
# 计数阵索引:0=n_cand 6=n_gold_pairs 9=fp_pair


def doc_na_density(arr):
    """(n,2) 文档级 density:fp_pair/(n_cand-n_gold_pairs),零除→0。"""
    denom = arr[:, :, 0] - arr[:, :, 6]
    num = arr[:, :, 9].astype(np.float64)
    return np.where(denom > 0, num / np.maximum(denom, 1), 0.0)


def main():
    docs = L.canonical_docs()
    n = len(docs)
    pos = {d: i for i, d in enumerate(docs)}
    w = bootweights(n, 10000, 20260926)
    reps = w.astype(np.float64)  # (B, n)

    golds_gD = defaultdict(dict)
    rel2idx = json.loads((PREP / "relation_to_index.json").read_text(encoding="utf-8"))
    for line in (PREP / "golds_gD.jsonl").read_text(encoding="utf-8-sig").split("\n"):
        if not line.strip():
            continue
        x = json.loads(line)
        rels = set(int(rel2idx[r]) for r in x["rels"] if r in rel2idx)
        if rels:
            golds_gD[x["doc_id"]][(x["h"], x["t"])] = rels
    cands = defaultdict(list)
    for line in (PREP / "candidates" / "evaluation.jsonl").read_text(encoding="utf-8-sig").split("\n"):
        if not line.strip():
            continue
        x = json.loads(line)
        cands[x["doc_id"]].append((x["h"], x["t"], x["group"]))

    # ---- 族 → 计数阵(own gD native)----
    fam_arrays = {}
    for gen in L.GEN_ORDER:
        g = L.GENERATORS[gen]
        name, own_gold = g["runs"][0]
        arrays = L.load_run_arrays(gen, name, [])
        # own gold 的 per_doc:run_file(gen,name,own)
        data = L.load_json(L.run_file(gen, name, own_gold))
        inner = data.get("per_doc", data)
        # 取 own/native 变体
        key = next((k for k in inner if k.endswith("|earliest_mention") and
                    ("native" in k.split("|")[2] or g["variants"][0] in k)), None)
        if key is None:
            continue
        e = inner[key]
        arr = np.zeros((n, 2, 10), np.int64)
        for i, dd in enumerate(e["docs"]):
            if dd not in pos:
                continue
            ii = pos[dd]
            for gi, side in enumerate(["intra", "inter"]):
                for z, ck in enumerate(L.CNT):
                    lst = e[side].get(ck)
                    if lst is not None and i < len(lst):
                        arr[ii, gi, z] = int(lst[i])
        fam_arrays[gen] = arr

    # 新族(三元组级重算)
    src = (Path(__file__).resolve().parent / "integrate_deberta_v3.py").read_text(encoding="utf-8")
    src = src.replace('sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")', 'pass')
    mod = _types.ModuleType("m")
    mod.__file__ = str(Path(__file__).resolve().parent / "integrate_deberta_v3.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    new_tags = [("deberta_v3", ROOT / "output_deberta_v3/s20261201/per_doc_native.json", True),
                ("longformer", ROOT / "output_longformer_compat/s20261201/per_doc_native.json", True),
                ("roberta_large", ROOT / "output_roberta_large/per_doc_native.json", True),
                ("bert_large", ROOT / "output_bert_large/per_doc_native.json", True),
                ("xlm_roberta_large", ROOT / "output_xlm_roberta_large/per_doc_native.json", True),
                ("electra_base", ROOT / "output_electra_base/per_doc_native.json", True),
                ("mdeberta_v3", ROOT / "output_mdeberta_v3/per_doc_native.json", True),
                ("ttmre_s74", ROOT / "out/ttmre_s74_fixed_decode.json", False),
                ("ttmre_s2", ROOT / "out/ttmre_s2_fixed_decode.json", False)]
    for tag, p, bridge in new_tags:
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        if bridge:
            preds = {did: {(h, r - 1, t) for h, r, t in tris} for did, tris in raw.items()}
        else:
            preds = {did: {(h, r, t) for h, r, t in tris} for did, tris in raw.items()}
        fam_arrays[tag] = mod.counts_for(preds, golds_gD, cands, docs, pos, n)

    print(f"族数: {len(fam_arrays)}")

    # ---- 双读数:F1 gap(原)与 NA density gap(新)----
    def f1(x):
        tp, npd, ngd = x[..., 3], x[..., 1], x[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((npd + ngd) > 0, 2 * tp / np.maximum(npd + ngd, 1e-12), 0.0)

    def boot_metric(arr, metric):
        """doc 级 metric 值 → 配对重采样均值差(intra-inter)。"""
        vals = metric(arr)  # (n,2)
        pooled = (reps @ vals) / reps.sum(1, keepdims=True)  # (B,2)
        return pooled[:, 0] - pooled[:, 1]

    fam_gaps = {}
    for tag, arr in fam_arrays.items():
        f1_gap = boot_metric(arr, lambda a: f1(a))
        na_gap = boot_metric(arr, doc_na_density)
        pt_f1 = float(np.mean(f1_gap))
        pt_na = float(np.mean(na_gap))
        fam_gaps[tag] = {"f1": {"point": pt_f1, "boot": f1_gap},
                         "na": {"point": pt_na, "boot": na_gap}}
        print(f"  {tag:22s} F1_gap={pt_f1:+.4f}  NA_gap={pt_na:+.4f}")

    # ---- 新轴翻转:gold(F1_vs_NA-margin) ----
    tags = list(fam_gaps.keys())
    flips = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            ti, tj = tags[i], tags[j]
            D1 = fam_gaps[ti]["f1"]["boot"] - fam_gaps[tj]["f1"]["boot"]
            D2 = fam_gaps[ti]["na"]["boot"] - fam_gaps[tj]["na"]["boot"]
            p1, p2 = float(np.mean(D1)), float(np.mean(D2))
            sg1, sg2 = np.sign(p1), np.sign(p2)
            if sg1 != sg2 and sg1 != 0 and sg2 != 0:
                FD = D1 - D2
                lo, hi = np.percentile(FD, [2.5, 97.5])
                flips.append({"pair": f"{ti}|{tj}", "dim": "gold(F1_vs_NA-margin)",
                              "D1_point": p1, "D2_point": p2, "flip": True,
                              "flip_delta_ci95": [float(lo), float(hi)],
                              "flip_delta_ci_excludes_zero": bool(lo > 0 or hi < 0)})
    robust = [f for f in flips if f["flip_delta_ci_excludes_zero"]]
    print(f"\nNA-margin 新轴翻转: {len(flips)}(robust {len(robust)})")
    for f in flips:
        print(f"  {f['pair']:38s} D1={f['D1_point']:+.4f} D2={f['D2_point']:+.4f} "
              f"CI排零={f['flip_delta_ci_excludes_zero']}")

    out = {"axis": "gold(F1_vs_NA-margin)", "families": tags,
           "flips": flips, "n_flip": len(flips), "n_robust": len(robust),
           "note": "NA spurious-pair density 端点(§7 NA-handling 争议);boot 10000×20260926"}
    (ROOT / "out" / "posthoc" / "na_margin_flips.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] out/posthoc/na_margin_flips.json")


if __name__ == "__main__":
    main()
