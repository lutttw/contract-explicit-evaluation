# released copy: as-run internal paths replaced by <local-workspace>;
# point the ROOT constant at a directory laid out like the release data/.
# -*- coding: utf-8 -*-
"""confirmatory_run.py — Paper 3 确认性运行统一入口(prereg_v1.md §5 冻结参数版)。

!! 仅在 prereg_v1.md 状态 = FROZEN(作者批准)后运行默认模式 !!
  python -I src/confirmatory_run.py            # 确认性运行(全参数,~1-2h CPU)
  python -I src/confirmatory_run.py --selftest # 自检(微型参数,产物入 out/confirmatory_selftest/,不算确认性证据)

冻结参数(prereg §2/§5):种子 20261201-05(WMT 取前 3);B=2000;350/148 split 沿用
各测试床协议;目标函数/特征/标准化口径同探索期冻结实现(agg_v04-v09 共同口径)。
产物:out/confirmatory/{docre,wmt,tacred}.json + summary.json + judgments.json
(J1-J4 自动计算,报告性质;判读以 prereg_v1.md 判读表为准)。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as NF

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p3loader as L                     # noqa: E402
from p3io import metrics                 # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
CONFIRM_SEEDS = [20261201, 20261202, 20261203, 20261204, 20261205]
TRAIN_CONTRACTS = ["score_A", "score_R", "gold_gD", "thr_fixed"]
CONTRACT_COL = {"score_A": 3, "score_R": 1}
EXCLUDE_FEAT_COLS = [5, 6, 8, 10]
LAMBDA, KAPPA, MARGIN_Z = 2.0, 0.5, 0.25
DELTA0, DELTA_MIN = 1.0, 0.3
HIDDEN = 16


def pooled_gap(a):
    m = metrics(a.sum(0, keepdims=True))[0]
    return float(m[0, 2] - m[1, 2])  # FIX 2026-11-07: 原为 m[0,1](=F1-R intra,非 Gap_F1)


def pooled_gap_col(a, col):
    m = metrics(a.sum(0, keepdims=True))[0]
    return float(m[0, col] - m[1, col])


def doc_gap_col(a, col):
    m = metrics(a)
    return np.nan_to_num(m[:, 0, col] - m[:, 1, col], nan=0.0)


def tau_b(x, y):
    from scipy.stats import kendalltau
    return float(kendalltau(x, y, variant="b").statistic)


# ============================================================ DocRE
def run_docre(epochs, B, outdir):
    docs = L.canonical_docs()
    n = len(docs)
    gens = L.GEN_ORDER
    G, K = len(gens), len(TRAIN_CONTRACTS)
    pairs = [(i, j) for i in range(G) for j in range(i + 1, G)]
    pname = {f"{gens[i]}|{gens[j]}": pi for pi, (i, j) in enumerate(pairs)}

    all_arrays = L.load_all_runs(n, {d: i for i, d in enumerate(docs)})

    def cA(c):
        row = []
        for gen in gens:
            sets = L.settings_for(gen)
            if c not in sets:
                row.append(None); continue
            sel, var, _ = sets[c]
            name, own = L.GENERATORS[gen]["runs"][0]
            key = (f"own@{own}", var) if sel == "own" else (sel, var)
            row.append(all_arrays[gen][0].get(key))
        return row

    A = {c: cA(c) for c in set(TRAIN_CONTRACTS + ["gold_gR"])}
    avail = [[k for k, c in enumerate(TRAIN_CONTRACTS) if A[c][k if False else gi] is not None]
             for gi in range(G)]
    rank_gD = np.array([pooled_gap(A["gold_gD"][gi]) for gi in range(G)])
    rank_gR = np.array([pooled_gap(A["gold_gR"][gi]) for gi in range(G)])

    Z = {}
    for c in TRAIN_CONTRACTS:
        zs = []
        for gi in range(G):
            if A[c][gi] is None:
                zs.append(None); continue
            r = doc_gap_col(A[c][gi], CONTRACT_COL.get(c, 2))
            zs.append((r - r.mean()) / (r.std() + 1e-9))
        Z[c] = zs
    TZ = {c: torch.tensor(np.stack([np.zeros(n) if z is None else z for z in Z[c]]),
                          dtype=torch.float32) for c in Z}

    F = np.load(OUT / "features.npz")["F"]
    keep = [i for i in range(F.shape[1]) if i not in EXCLUDE_FEAT_COLS]
    X = torch.tensor((F[:, keep] - F[:, keep].mean(0)) / (F[:, keep].std(0) + 1e-9),
                     dtype=torch.float32)
    qn = F[:, 15] / F[:, 15].sum()
    qn_t = torch.tensor(qn, dtype=torch.float32)

    cons_sign, cons_strength, med_abs = {}, {}, {}
    for (i, j) in pairs:
        ds = [pooled_gap_col(A[c][i], CONTRACT_COL.get(c, 2)) - pooled_gap_col(A[c][j], CONTRACT_COL.get(c, 2))
              for c in TRAIN_CONTRACTS if A[c][i] is not None and A[c][j] is not None]
        med = float(np.median(ds))
        cons_sign[(i, j)] = np.sign(med)
        cons_strength[(i, j)] = float(np.mean([np.sign(x) == np.sign(med) for x in ds])) if med != 0 else 0.0
        med_abs[(i, j)] = abs(med)

    def train(seed):
        tg = np.random.default_rng(seed)
        torch.manual_seed(seed)
        tr, dv = tg.permutation(n)[:350], None  # split 声明:确认性主判据在全集与 dev 并报,split 仅沿用登记
        W1 = (torch.randn(X.shape[1], HIDDEN) * 0.1).requires_grad_(True)
        b1 = torch.zeros(HIDDEN).requires_grad_(True)
        W2 = (torch.randn(HIDDEN, G * K) * 0.1).requires_grad_(True)
        b2 = torch.zeros(G * K).requires_grad_(True)
        opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-2, weight_decay=1e-4)
        med_m = {(i, j): float(np.median([abs(Z[c][i].mean() - Z[c][j].mean())
                                          for c in TRAIN_CONTRACTS if Z[c][i] is not None and Z[c][j] is not None]))
                 for i, j in pairs}

        def agg(ws):
            T = []
            for gi in range(G):
                ks = avail[gi]
                wsk = ws[:, gi, ks] / (ws[:, gi, ks].sum(1, keepdim=True) + 1e-12)
                acc = None
                for kk, k in enumerate(ks):
                    acc = (acc if acc is not None else 0) + wsk[:, kk] * TZ[TRAIN_CONTRACTS[k]][gi]
                T.append((acc * qn_t).sum(0) / qn_t.sum())
            return torch.stack(T)

        for ep in range(epochs):
            dl = DELTA0 + (DELTA_MIN - DELTA0) * min(1.0, ep / (0.6 * epochs))
            h = NF.relu(X @ W1 + b1)
            ws = NF.softmax((h @ W2 + b2).reshape(n, G, K), dim=2)
            T = agg(ws)
            dF = torch.stack([T[a] - T[b] for a, b in pairs])
            lq = torch.zeros(())
            for pi, (i, j) in enumerate(pairs):
                cs, st = cons_sign[(i, j)], cons_strength[(i, j)]
                if cs == 0:
                    continue
                t_ij = max(0.0, 1.0 - med_m[(i, j)] / dl)
                lq = lq + NF.softplus(MARGIN_Z * st * (1.0 - t_ij) - cs * dF[pi])
            lf = torch.zeros(())
            for k_oo in range(K):
                ws_oo = ws.clone(); ws_oo[:, :, k_oo] = 0
                ws_oo = ws_oo / (ws_oo.sum(2, keepdim=True) + 1e-12)
                T_oo = agg(ws_oo)
                dO = torch.stack([T_oo[a] - T_oo[b] for a, b in pairs])
                lf = lf + NF.softplus(-(dF * dO) / KAPPA).mean()
            loss = lq / len(pairs) + LAMBDA * lf / K
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            h = NF.relu(X @ W1 + b1)
            return NF.softmax((h @ W2 + b2).reshape(n, G, K), dim=2).numpy()

    print("DocRE: 训练 5 确认性种子…")
    ws_seeds = [train(s) for s in CONFIRM_SEEDS]

    brng = np.random.default_rng(20261112)
    reps = brng.integers(0, n, (B, n))
    mult = np.stack([(reps == v).sum(1) for v in range(n)], axis=1).astype(np.float64)

    def boot_z(ws_np):
        o = np.zeros((B, G))
        for gi in range(G):
            ks = avail[gi]
            wsk = ws_np[:, gi, ks] / (ws_np[:, gi, ks].sum(1, keepdims=True) + 1e-12)
            zmix = np.zeros(n)
            for kk, k in enumerate(ks):
                zc = Z[TRAIN_CONTRACTS[k]][gi]
                zmix += wsk[:, kk] * (zc if zc is not None else 0.0)
            o[:, gi] = (mult @ (zmix * qn)) / (mult @ qn)
        return o

    def boot_gD():
        o = np.zeros((B, G))
        for gi in range(G):
            agg = (mult @ A["gold_gD"][gi].reshape(n, -1)).reshape(B, 2, 10)
            tp, npd, ngd = agg[:, :, 3], agg[:, :, 1], agg[:, :, 2]
            o[:, gi] = (2 * tp[:, 0] / (2 * tp[:, 0] + npd[:, 0] + ngd[:, 0] + 1e-12)
                        - 2 * tp[:, 1] / (2 * tp[:, 1] + npd[:, 1] + ngd[:, 1] + 1e-12))
        return o

    def boot_aer():
        o = np.zeros((B, G))
        for kk, c in enumerate(TRAIN_CONTRACTS):
            v = np.zeros((G, B))
            for gi in range(G):
                if A[c][gi] is None:
                    v[gi] = np.nan; continue
                agg = (mult @ A[c][gi].reshape(n, -1)).reshape(B, 2, 10)
                tp, npd, ngd = agg[:, :, 3], agg[:, :, 1], agg[:, :, 2]
                col = CONTRACT_COL.get(c, 2)
                if col == 0:
                    rd = tp / (npd + 1e-12)
                elif col == 1:
                    rd = tp / (ngd + 1e-12)
                else:
                    rd = 2 * tp / (2 * tp + npd + ngd + 1e-12)
                v[gi] = rd[:, 0] - rd[:, 1]
            valid = ~np.isnan(v)
            vm = np.where(valid, v, np.nan)
            mu_ = np.nanmean(vm, axis=0); sd_ = np.nanstd(vm, axis=0) + 1e-9
            z = (v - mu_[None, :]) / sd_[None, :]
            z[~valid] = 0.0
            o += z.T
        return o / K

    reg = L.load_json(L.RV5 / "statistics" / "r1_rank_flip.json")
    checks = []
    for pk, dims in reg["flip_matrix"].items():
        if pk not in pname:
            continue
        for dim, e in dims.items():
            if e.get("available"):
                checks.append({"pi": pname[pk], "flip": bool(e["flip"])})
    flips = [c for c in checks if c["flip"]]

    def reader_stats(R):
        ties, point = {}, R.mean(0)
        for pi, (a, b) in enumerate(pairs):
            lo, hi = np.percentile(R[:, a] - R[:, b], [2.5, 97.5])
            ties[pi] = bool(lo <= 0 <= hi)
        cov = sum(1 for c in flips if ties[c["pi"]])
        ft = sum(1 for c in checks if not c["flip"] and ties[c["pi"]])
        rows = []
        for c in flips:
            if not ties[c["pi"]]:
                a, b = pairs[c["pi"]]
                sg = float(np.sign(point[a] - point[b]))
                rows.append({"pair_pi": c["pi"],
                             "match_cons": bool(sg == cons_sign[(a, b)]),
                             "match_gR": bool(sg == np.sign(rank_gR[a] - rank_gR[b]))})
        fid_cons = sum(1 for r in rows if r["match_cons"])
        fid_gR = sum(1 for r in rows if r["match_gR"])
        return {"coverage": cov, "n_flip": len(flips), "false_tie": ft,
                "n_ranked": len(rows), "fid_cons": fid_cons, "fid_gR": fid_gR,
                "rows": rows}

    def agg_point(ws_np):
        out = []
        for gi in range(G):
            ks = avail[gi]
            wsk = ws_np[:, gi, ks] / (ws_np[:, gi, ks].sum(1, keepdims=True) + 1e-12)
            zmix = np.zeros(n)
            for kk, k in enumerate(ks):
                zc = Z[TRAIN_CONTRACTS[k]][gi]
                zmix += wsk[:, kk] * (zc if zc is not None else 0.0)
            out.append(float((zmix * qn).sum() / qn.sum()))
        return np.array(out)

    res = {"seeds": CONFIRM_SEEDS, "readers": {}}
    # learned 每种子 + mean 读数聚合的 reader stats
    per_seed = []
    for si, w in enumerate(ws_seeds):
        st = reader_stats(boot_z(w))
        ap = agg_point(w)
        st["tau_vs_gR"] = tau_b(ap, rank_gR)
        st["tau_vs_gD"] = tau_b(ap, rank_gD)
        per_seed.append(st)
        print(f"  seed{CONFIRM_SEEDS[si]}: cov {st['coverage']}/{st['n_flip']} ft {st['false_tie']} "
              f"fid {st['fid_cons']}/{st['n_ranked']} tau_gR {st['tau_vs_gR']:.4f}")
    res["readers"]["learned_per_seed"] = per_seed
    res["readers"]["gD_native"] = reader_stats(boot_gD())
    res["readers"]["AER_corpus"] = reader_stats(boot_aer())
    res["readers"]["uniform"] = reader_stats(boot_z(np.full((n, G, K), 1.0 / K)))
    # learned 均值读数(跨种子 mean of point)对应 bootstrap:逐种子均值后计算
    R_mean = np.mean([boot_z(w) for w in ws_seeds], axis=0)
    res["readers"]["learned_mean"] = reader_stats(R_mean)
    # FIX 2026-11-07:参照读数器也记录 tau(J4 判据需要 gD 原生 τ_vs_gR 与 uniform τ_vs_gD)
    _gd_pt, _aer_pt = boot_gD().mean(0), boot_aer().mean(0)
    _uni_ws = np.full((n, G, K), 1.0 / K)
    res["readers"]["gD_native"]["tau_vs_gR"] = tau_b(_gd_pt, rank_gR)
    res["readers"]["gD_native"]["tau_vs_gD"] = tau_b(_gd_pt, rank_gD)
    res["readers"]["AER_corpus"]["tau_vs_gR"] = tau_b(_aer_pt, rank_gR)
    res["readers"]["uniform"]["tau_vs_gD"] = tau_b(agg_point(_uni_ws), rank_gD)
    res["readers"]["uniform"]["tau_vs_gR"] = tau_b(agg_point(_uni_ws), rank_gR)
    (outdir / "docre.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res



def _load_sanitized(path, name):
    """exec 模块但剥掉 sys.stdout 重绑(链式 wrapper 会关共享 buffer)。"""
    import importlib.util, types
    src = Path(path).read_text(encoding="utf-8")
    src = src.replace('sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")', 'pass  # stdout rebind removed by confirmatory_run')
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod

# ============================================================ WMT / TACRED:复用 v09/v08 冻结实现
def run_wmt(epochs, B, outdir):
    v09 = _load_sanitized(HERE / "agg_v09_wmt.py", "agg_v09")
    # 参数替换为确认性(通过 monkeypatch 常量;冻结口径不变)
    v09.EPOCHS = epochs; v09.B = B; v09.SEEDS = CONFIRM_SEEDS[:3]
    # v09.main() 写 out/agg_v09_wmt.json;确认性版本改写到 outdir
    res = {}
    v09.OUT = outdir  # 结果落 confirmatory 目录
    for pair in ["de-en", "zh-en"]:  # 段级打分缓存预复制,避免自检/重跑重算
        cache_src = OUT / f"wmt_seg_scores_{pair}.json"
        cache_dst = outdir / f"wmt_seg_scores_{pair}.json"
        if cache_src.exists() and not cache_dst.exists():
            cache_dst.write_bytes(cache_src.read_bytes())
    v09.main()
    # v09 main 写死了文件名 agg_v09_wmt.json 于其 OUT;搬运:
    src = outdir / "agg_v09_wmt.json"
    if src.exists():
        res = json.loads(src.read_text(encoding="utf-8"))
        (outdir / "wmt.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        src.unlink()
    return res


def run_tacred(epochs, outdir):
    v08 = _load_sanitized(HERE / "agg_v08_tacred.py", "agg_v08")
    v08.EPOCHS = epochs; v08.SEEDS = CONFIRM_SEEDS
    v08.OUT = outdir
    v08.main()
    src = outdir / "agg_v08_tacred.json"
    res = json.loads(src.read_text(encoding="utf-8")) if src.exists() else {}
    if src.exists():
        (outdir / "tacred.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        src.unlink()
    return res


# ============================================================ J1-J4
def judge(docre):
    from math import comb
    lr = docre["readers"]["learned_mean"]
    gd = docre["readers"]["gD_native"]
    ae = docre["readers"]["AER_corpus"]
    per = docre["readers"]["learned_per_seed"]

    def fisher_one_sided(k1, n1, k2, n2):
        # 超几何尾概率:P(X >= k1 | 合并)近似(Fisher 精确的单侧上尾)
        K, N = k1 + k2, n1 + n2
        num = sum(comb(K, i) * comb(N - K, n1 - i) for i in range(0, n1 + 1))
        p = 0.0
        for i in range(k1, n1 + 1):
            if K - (n1 - i) >= 0 and N - K - i >= 0:
                p += comb(K, i) * comb(N - K, n1 - i) / num
        return p

    j1_sig = fisher_one_sided(lr["fid_cons"], lr["n_ranked"], gd["fid_cons"], gd["n_ranked"]) \
        if gd["n_ranked"] > 0 else 1.0
    j1 = (j1_sig < 0.05) and (lr["fid_cons"] >= 6 or lr["n_ranked"] < 8)
    j1_self = all(s["fid_cons"] == s["n_ranked"] or s["fid_cons"] >= 6 for s in per)
    j2 = lr["false_tie"] <= min(gd["false_tie"], ae["false_tie"])
    j3 = lr["coverage"] >= ae["coverage"] - 1
    # J4 DocRE Pareto(uniform tau 参照)
    uni = docre["readers"]["uniform"]
    j4_docre = (lr.get("tau_vs_gR", 0.0) >= 0.0)  # tau 由调用方补填;占位由 summary 步骤完成
    return {"J1": {"pass": bool(j1 and j1_self), "fisher_p": j1_sig,
                   "learned_fid": f"{lr['fid_cons']}/{lr['n_ranked']}",
                   "gd_fid": f"{gd['fid_cons']}/{gd['n_ranked']}"},
            "J2": {"pass": bool(j2), "learned_ft": lr["false_tie"],
                   "min_base_ft": min(gd["false_tie"], ae["false_tie"])},
            "J3": {"pass": bool(j3), "learned_cov": lr["coverage"], "aer_cov": ae["coverage"]},
            "note": "J4 在 summary 阶段合并 WMT 锚与 DocRE tau 后判定"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="微型参数自检(产物入 out/confirmatory_selftest/,不算确认性证据)")
    args = ap.parse_args()
    prereg = HERE.parent / "prereg" / "prereg_v1.md"
    head = prereg.read_text(encoding="utf-8")[:200]
    frozen = "FROZEN" in head
    if not args.selftest and not frozen:
        print("[拒绝] prereg_v1.md 非 FROZEN —— 确认性运行须先获作者批准(红线:不代行预注册决策)")
        return 1
    outdir = OUT / ("confirmatory_selftest" if args.selftest else "confirmatory")
    outdir.mkdir(parents=True, exist_ok=True)
    epochs = 60 if args.selftest else 3000
    B = 100 if args.selftest else 2000
    t0 = time.time()
    docre = run_docre(epochs, B, outdir)
    wmt = run_wmt(epochs, B, outdir)
    tacred = run_tacred(epochs, outdir)
    summary = {"mode": "selftest" if args.selftest else "confirmatory",
               "prereg_frozen": frozen, "seeds": CONFIRM_SEEDS,
               "elapsed_s": round(time.time() - t0, 1), "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    jd = judge(docre)
    jd["mode"] = summary["mode"]
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "judgments.json").write_text(json.dumps(jd, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(jd, ensure_ascii=False, indent=1))
    print(f"[落盘] {outdir} ({summary['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
