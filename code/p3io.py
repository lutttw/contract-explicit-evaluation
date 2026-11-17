# released copy: as-run internal paths replaced by <local-workspace>;
# point the ROOT constant at a directory laid out like the release data/.
# -*- coding: utf-8 -*-
"""p3io.py — Paper 3 读数协议库(自含重实现)。

复现门要求 Paper 3 用自己的 loader 重建冻结读数。本模块按
04_文档级IE/docs/src/jcip_audit/rev5_recount_lib.py 的冻结公式逐行重实现
(只读参照,不 import),公式一致性由双门逐位对照保证:

  CNT/MET 常数、metrics()、bootweights()、boot_gaps()、to_canonical()。

红线:所有输入为 04_文档级IE 冻结资产,只读;26+ 冻结 run 永不重训——
本库全部工作发生在"读数层"(文档级计数的聚合),不触预测生成层。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 冻结常数(rev5_recount_lib 逐字)
CNT = ['n_cand', 'n_pred', 'n_gold', 'tp', 'fp', 'fn', 'n_gold_pairs', 'hit', 'exact', 'fp_pair']
MET = ['P', 'R', 'F1', 'A', 'E']
COL = {m: j for j, m in enumerate(MET)}


def load_json(p) -> dict:
    """rev5 口径:utf-8-sig 读入。"""
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def metrics(a):
    """计数 → [P, R, F1, A, E];池化口径,F1 = 2TP/(n_pred+n_gold),零除→nan。"""
    k = a[..., 2]; l = a[..., 1]; h = a[..., 3]; q = a[..., 6]

    def div(x, y):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(y > 0, x / y, np.nan)
    return np.stack([div(h, l), div(h, k), div(2 * h, k + l), div(a[..., 7], q),
                     div(a[..., 8], q)], axis=-1)


def bootweights(n_docs, reps, seed):
    """文档级配对自助的 (reps, n_docs) 权重矩阵(同一次重采样跨臂共用)。"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_docs, (reps, n_docs))
    w = np.zeros((reps, n_docs), np.int16)
    for b, row in enumerate(idx):
        w[b] = np.bincount(row, minlength=n_docs)
    return w


def boot_gaps(a, w):
    """逐文档计数 (n_docs,2,10) 经权重聚合后逐 replicate 的组内−跨句 gap(5 指标)。"""
    acc = (w.astype(np.float64) @ a.reshape(len(a), -1)).reshape(len(w), 2, len(CNT))
    bm = metrics(acc)
    return bm[:, 0] - bm[:, 1]


def to_canonical(docs, intra, inter, canon_pos, n_docs):
    """把一个 run 的逐文档计数重排到规范文档序,返回 (n_docs, 2, 10) 计数阵。"""
    a = np.zeros((n_docs, 2, len(CNT)), np.int64)
    idx = np.array([canon_pos[d] for d in docs])
    for g, src in ((0, intra), (1, inter)):
        for z, k in enumerate(CNT):
            a[idx, g, z] = np.asarray(src[k], dtype=np.int64)
    assert a[:, :, 3].sum() + a[:, :, 4].sum() == a[:, :, 1].sum(), "tp+fp≠n_pred(装载错位)"
    return a


def load_counts(path, key: str):
    """读一个 (金标,解码配置,分组规则) 的逐文档计数;返回 (docs, intra, inter)。

    X2/P3M 的导出顶层即配置键;ATLOP 系的导出为 {run_id, ts, per_doc:{配置键}},需剥一层。
    """
    d = load_json(path)
    if key not in d and "per_doc" in d:
        d = d["per_doc"]
    if key not in d:
        raise SystemExit(f"[缺键] {Path(path).name} 缺 {key}(可用:{sorted(d)[:4]}…)")
    e = d[key]
    return e["docs"], e["intra"], e["inter"]


def deep_compare(got, want, path="", out=None, float_tol=0.0):
    """递归对照两个 JSON 形结构;float_tol=0 即逐位一致判据。返回差异清单。"""
    if out is None:
        out = []
    if isinstance(want, dict) and isinstance(got, dict):
        for k in sorted(set(want) | set(got)):
            if k not in want:
                out.append({"path": f"{path}.{k}", "kind": "extra_in_got"})
            elif k not in got:
                out.append({"path": f"{path}.{k}", "kind": "missing_in_got"})
            else:
                deep_compare(got[k], want[k], f"{path}.{k}", out, float_tol)
    elif isinstance(want, list) and isinstance(got, list):
        if len(got) != len(want):
            out.append({"path": path, "kind": "len", "got": len(got), "want": len(want)})
        for i, (g, w) in enumerate(zip(got, want)):
            deep_compare(g, w, f"{path}[{i}]", out, float_tol)
    elif isinstance(want, bool) or isinstance(got, bool):
        if got is not want:
            out.append({"path": path, "kind": "value", "got": got, "want": want})
    elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
        if float(got) != float(want):
            out.append({"path": path, "kind": "float" if float_tol > 0 else "value",
                        "got": got, "want": want,
                        "absdiff": abs(float(got) - float(want))})
    elif got != want:
        out.append({"path": path, "kind": "value", "got": got, "want": want})
    return out
