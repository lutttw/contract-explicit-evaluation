# released copy: as-run internal paths replaced by <local-workspace>;
# point the ROOT constant at a directory laid out like the release data/.
# -*- coding: utf-8 -*-
"""p3loader.py — Paper 3 DocRE 数据装载器。

把 04_文档级IE 冻结资产里 7 生成器 × 44 run 的逐文档预测计数,统一装载为
规范 498 文档序的 (n_docs, 2, 10) 计数阵,并产出学习侧统一记录:

  - 每生成器 × 契约设置(6 端:score_A/score_R/gold_gD/gold_gR/thr_fixed/thr_cal)
    的逐 run 计数阵与文档元数据(doc_len、双金标关系频率);
  - readings():聚合器特征原料(逐文档 tp/fp/fn/n_pred/n_cand + 池化读数)。

生成器清单/契约键/文档序全部按冻结脚本 r1_rank_flip.py 的口径镜像;
逐位正确性由 repro_docre.py(门 A)对 r1_rank_flip.json + r64_ec1 的硬对照保证。

红线:输入全部只读;本模块不生成任何预测。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p3io import CNT, COL, boot_gaps, load_counts, load_json, metrics, to_canonical

# ---- 冻结资产根(只读)----
ROOT = Path(r"<local-workspace>")
RUNS = ROOT / "out" / "jcip_audit_v1" / "runs"        # X2 多标签 runs
RV5 = ROOT / "out" / "revision_v5_experiments"        # ATLOP 两版 + statistics
X3B = ROOT / "out" / "jcip_audit_v1" / "x3_bridged"   # P3M 桥接 per_doc
DOCRE_DATA = ROOT / "data" / "docred"                 # dev.json(gD)/ dev_revised.json(gR)

RULE = "earliest_mention"                # 分组规则冻结
X2_SEEDS = ("20260917", "20260918", "20260919", "20261001", "20261002")
ATLOP_SEEDS = ("20260917", "20260918", "20260919")
P3M_SEEDS = ("20260917", "20260918", "20260919", "20261001", "20261002")

# rev5 库 MET 列序 = ['P','R','F1','A','E'];计分契约两端列:A(hit-any)/ R(标签级 recall)
SCORE_COL = {"A": COL["A"], "R": COL["R"]}
GAP_COL = COL["F1"]                      # 排名量 = Gap_F1(句内−跨句)


def spec_x2(gold, variant):
    return f"evaluation|{gold}|{variant}|sigmoid_multilabel|{RULE}"


def spec_atlop(gold, variant):
    return f"evaluation|{gold}|{variant}|{RULE}"


def spec_p3m(gold, variant):
    return f"evaluation|{gold}|{variant}|p3m_native|{RULE}"


def x2_runs():
    """X2 全 20 run:(run_id, 自身金标)——D 臂→gD,R 臂→gR。"""
    out = []
    for r in ("r025", "r100"):
        for arm in ("D", "R"):
            for s in X2_SEEDS:
                out.append((f"ml_{arm}_{r}_s{s}", "gD" if arm == "D" else "gR"))
    return out


def atlop_runs(base: Path):
    """ATLOP 单版 6 run:(文件名, 自身金标)。"""
    out = []
    for arm in ("D", "R"):
        for s in ATLOP_SEEDS:
            out.append((f"atlop_{arm}_s{s}", "gD" if arm == "D" else "gR"))
    for name, _ in out:
        if not (base / f"per_doc_{name}.json").exists():
            raise SystemExit(f"[缺文件] {base / ('per_doc_' + name + '.json')}")
    return out


def find_per_doc(run_dir: Path, gold: str) -> Path:
    """X2 run 的逐文档导出:早期 run 在 per_doc/ 子目录,扩展种子 run 在 run 根目录。"""
    for p in (run_dir / "per_doc" / f"evaluation__{gold}.json",
              run_dir / f"evaluation__{gold}.json"):
        if p.exists():
            return p
    raise SystemExit(f"[缺文件] {run_dir} 下既无 per_doc/evaluation__{gold}.json "
                     f"也无 evaluation__{gold}.json")


# 生成器注册表(与冻结脚本 r1_rank_flip.py GENERATORS 同序同义)
GENERATORS = {
    "x2": {
        "label": "X2 多标签(DocRED,20 run)",
        "runs": x2_runs(),
        "spec_fn": spec_x2,
        "variants": ("fixed_0.5", "cal_sigmoid_multilabel"),
        "has_thr": True,
        "native_gold": "own",
    },
    "atlop_pe8": {
        "label": "ATLOP PE8 重训(校准分片排除,3 种子)",
        "runs": atlop_runs(RV5 / "independent_model_retrain"),
        "base": RV5 / "independent_model_retrain",
        "spec_fn": spec_atlop,
        "variants": ("atlop_native", "atlop_sigmoid05_ref"),
        "has_thr": True,
        "native_gold": "own",
    },
    "atlop_full": {
        "label": "ATLOP 首轮全量训练变体(3 种子)",
        "runs": atlop_runs(RV5 / "independent_model"),
        "base": RV5 / "independent_model",
        "spec_fn": spec_atlop,
        "variants": ("atlop_native", "atlop_sigmoid05_ref"),
        "has_thr": True,
        "native_gold": "own",
    },
    "dreeam": {
        "label": "DREEAM PE8 自训(公开方法,校准分片排除,1 种子)",
        "runs": [(f"dreeam_pe8_s{s}_l01", "gD") for s in ATLOP_SEEDS],
        "base": RV5 / "dreeam",
        "spec_fn": spec_atlop,
        "variants": ("atlop_native", "atlop_sigmoid05_ref"),
        "has_thr": True,
        "native_gold": "own",
    },
    "p3m": {
        "label": "P3M 原生决策规则桥接(5 种子)",
        "runs": [(f"per_doc_x3_p3m_s{s}", "gD") for s in P3M_SEEDS],
        "base": X3B,
        "spec_fn": spec_p3m,
        "variants": ("native", None),
        "has_thr": False,
        "native_gold": "gD",
    },
    "docunet": {
        "label": "DocuNet PE8 自训(公开方法-分割视角,校准分片排除,1 种子)",
        "runs": [(f"docunet_pe8_s{s}", "gD") for s in ATLOP_SEEDS[:2]],
        "base": RV5 / "docunet",
        "spec_fn": spec_atlop,
        "variants": ("atlop_native", "atlop_sigmoid05_ref"),
        "has_thr": True,
        "native_gold": "own",
    },
    "kddocre": {
        "label": "KD-DocRE PE8 自训(公开方法-轴向注意力视角,校准分片排除,1 种子)",
        "runs": [(f"kddocre_pe8_s{s}", "gD") for s in ATLOP_SEEDS[:2]],
        "base": RV5 / "kddocre",
        "spec_fn": spec_atlop,
        "variants": ("atlop_native", "atlop_sigmoid05_ref"),
        "has_thr": True,
        "native_gold": "own",
    },
}

GEN_ORDER = list(GENERATORS.keys())      # 冻结顺序:x2, atlop_pe8, atlop_full, dreeam, p3m, docunet, kddocre

# 契约设置(生成器 × 契约维度 × 两端)→ (gold 来源, variant, 指标列)
def settings_for(gen):
    """返回 {设置名: (gold_sel, variant, col)};gold_sel ∈ {'own','gD','gR'}。"""
    g = GENERATORS[gen]
    v = g["variants"]
    s = {
        "score_A": ("own", v[0], SCORE_COL["A"]),
        "score_R": ("own", v[0], SCORE_COL["R"]),
        "gold_gD": ("gD", v[0], GAP_COL),
        "gold_gR": ("gR", v[0], GAP_COL),
    }
    if g["has_thr"]:
        s["thr_fixed"] = ("own", v[0], GAP_COL)
        s["thr_cal"] = ("own", v[1], GAP_COL)
    return s


CONTRACT_DIMS = {
    "scoring(A_vs_R)": ("score_A", "score_R"),
    "gold(gD_vs_gR)": ("gold_gD", "gold_gR"),
    "threshold(fixed_vs_cal)": ("thr_fixed", "thr_cal"),
}


def run_file(gen, name, gold):
    """单个 run × 单金标的 per_doc 文件路径。"""
    g = GENERATORS[gen]
    if gen == "x2":
        return find_per_doc(RUNS / name, gold)
    if gen == "p3m":
        return g["base"] / name / f"evaluation__{gold}.json"
    return g["base"] / f"per_doc_{name}.json"


def load_run_arrays(gen, name, golds_needed):
    """装载一个 run 在给定金标集合下的全部 (gold, variant) 计数(规范序前原始形态)。"""
    g = GENERATORS[gen]
    out = {}
    docs0 = None
    for gold in golds_needed:
        p = run_file(gen, name, gold)
        for variant in ({v for (_, v, _) in settings_for(gen).values()} - {None}):
            docs, intra, inter = load_counts(p, g["spec_fn"](gold, variant))
            if docs0 is None:
                docs0 = docs
            elif docs != docs0:
                raise SystemExit(f"[序错] {gen}/{name} {gold}: 同文件不同配置文档序不一致")
            out[(gold, variant)] = (docs, intra, inter)
    return out


def canonical_docs():
    """规范文档序(取自 X2 评测总体,sorted 后 498 篇)。"""
    probe = load_json(find_per_doc(RUNS / "ml_D_r025_s20260917", "gD"))
    docs_all = sorted(probe["evaluation|gD|fixed_0.5|sigmoid_multilabel|" + RULE]["docs"])
    return docs_all


def load_all_runs(n_docs, canon_pos, golds_by_gen=None):
    """装载全部 44 run 的全部所需契约计数阵(规范序)。

    返回 arrays[gen][(gold_sel_effective, variant)] 列表(逐 run),
    其中键与 r1_rank_flip.py 的 'own@{gold}' 约定一致:
      arrays[gen] = [{(key1,variant): ndarray(n_docs,2,10), ...}, ...per run]
    golds_by_gen: 可选覆盖 {gen: [golds]},默认由 settings 推导。
    """
    all_arrays = {}
    for gen in GEN_ORDER:
        g = GENERATORS[gen]
        sets = settings_for(gen)
        golds_needed = sorted({sel for (sel, _, _) in sets.values()} - {None})
        if golds_by_gen and gen in golds_by_gen:
            golds_needed = golds_by_gen[gen]
        runs_arrays = []
        for name, own_gold in g["runs"]:
            cross_golds = [gd for gd in golds_needed if gd != "own"]
            arrays = load_run_arrays(gen, name, cross_golds)
            own_arr = load_run_arrays(gen, name, [own_gold])
            arrays.update({(f"own@{own_gold}", k[1]): v for k, v in own_arr.items()})
            canon = {}
            for k, (docs, intra, inter) in arrays.items():
                canon[k] = to_canonical(docs, intra, inter, canon_pos, n_docs)
            runs_arrays.append(canon)
        all_arrays[gen] = runs_arrays
    return all_arrays


# ---------------------------------------------------------------- 文档元数据

def doc_metadata(docs_all):
    """学习侧统一记录:每文档 {doc_id(=title 键), n_gold_gD/gR, rel_freq_gD/gR}。

    注:评测 doc_id 为哈希空间(prepared/ 口径),与 dev.json 的 title 字段
    不直接对应——句级 doc_len 的映射留到 Day 3 特征日再解;本函数先给出
    与读数同空间的文档级特征(n_gold、逐关系频率、intra/inter 结构)。
    """
    prepared = ROOT / "out" / "lre_rebuttal_v2" / "prepared"
    rel2id = load_json(prepared / "relation_to_index.json")
    n_rel = len(rel2id)

    def read_golds(name):
        gold = {}
        for line in (prepared / name).read_text(encoding="utf-8-sig").split("\n"):
            if line.strip():
                x = json.loads(line)
                # golds 的 rels 为关系名(P580 式),经 rel2id 转冻结索引
                gold.setdefault(x["doc_id"], []).extend(int(rel2id[r]) for r in x["rels"])
        return gold

    gD = read_golds("golds_gD.jsonl")
    gR = read_golds("golds_gR.jsonl")
    meta = []
    for d in docs_all:
        rg, rr = gD.get(d, []), gR.get(d, [])
        fD, fR = np.zeros(n_rel, np.int64), np.zeros(n_rel, np.int64)
        for r in rg:
            fD[r] += 1
        for r in rr:
            fR[r] += 1
        meta.append({"doc_id": d, "n_gold_gD": len(rg), "n_gold_gR": len(rr),
                     "rel_freq_gD": fD, "rel_freq_gR": fR})
    return meta, rel2id


# ---------------------------------------------------------------- 读数聚合

def compute_gaps(all_arrays, n_docs, w, gens=None):
    """r1_rank_flip.py 主计算镜像:每生成器 × 契约设置的 gap(点估计 + 重采样分布)。

    返回 (gaps, density):
      gaps[gen][setting] = {'point':…, 'ci95':[lo,hi], 'per_run_points':[…], '_boot':(reps,)}
      density[gen] = ∑n_pred/∑n_cand(跨 run 平均,own 金标 × 默认变体)
    """
    gaps, density = {}, {}
    for gen in (gens or GEN_ORDER):
        g = GENERATORS[gen]
        sets = settings_for(gen)
        boots = {s: [] for s in sets}
        pts = {s: [] for s in sets}
        dens_runs, run_points = [], {}
        for (name, _run_gold), canon in zip(g["runs"], all_arrays[gen]):
            own_gold_keys = [k for k in canon if k[0].startswith("own@")]
            own_gold = own_gold_keys[0][0] if own_gold_keys else None
            for s, (sel, variant, col) in sets.items():
                key = (own_gold, variant) if sel == "own" else (sel, variant)
                a = canon[key]
                bg = boot_gaps(a, w)
                mt = metrics(a.sum(0, keepdims=True))[0]
                boots[s].append(bg[:, col])
                pts[s].append(float(mt[0, col] - mt[1, col]))
                run_points.setdefault(s, []).append(
                    {"run": name, "gold_used": own_gold.split("own@")[-1] if sel == "own" else sel,
                     "point": float(mt[0, col] - mt[1, col])})
            key0 = (own_gold, g["variants"][0] if g["variants"] else None)
            a0 = canon[key0]
            dens_runs.append(float(a0[:, :, 1].sum()) / float(a0[:, :, 0].sum()))
        gaps[gen] = {}
        for s in sets:
            boot = np.mean(boots[s], axis=0)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            gaps[gen][s] = {"point": float(np.mean(pts[s])), "ci95": [float(lo), float(hi)],
                            "per_run_points": run_points[s], "_boot": boot}
        density[gen] = float(np.mean(dens_runs))
    return gaps, density
