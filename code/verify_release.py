# -*- coding: utf-8 -*-
"""verify_release.py — Re-derive every headline number of the paper from the
released frozen artifacts alone (data/ + judge_raw). CPU-minutes; numpy/scipy.

Run:  python code/verify_release.py        (from the repository root)

Checks
  V1  judge_raw integrity: 17 judges, 848,946 records, required fields
  V2  census re-derivation (prereg v7 SPLIT_CRIT=0.6 majority-rate rule):
      n=49,667 battles judged, contested share 12.25%, 6,083 contested
  V3  Clopper-Pearson one-sided (90%) upper bounds: census 0.1244,
      v6 DocRE census 0.1456, stratum tiers high/mid/low
  V4  cross-file consistency: stratum overall == census; high+mid tier size
      == certified-battle count in rank stability == cluster bootstrap
  V5  E2 exact McNemar: b=7, c=0 -> p = 2 * P[X<=0; Bin(7, 0.5)] = 0.0156
  V6  E3 retroactive probe: 9 threshold flips, 2 persist -> 7 resolve;
      x2|docunet excluded (96-column space, registered for v8)
  V7  E5 near-tie stratification shares + corrected Krippendorff alpha
      (3-class 0.4561, direction-only 0.5035); stale first-pass alpha
      (-0.0878, wrong expected-disagreement term) must be absent
  V8  E7 Bradley-Terry rank stability: tau 0.9977, cert+hedge = 49,938
  V9  NA-margin endpoint: 69/69 pairs flip on the gold axis
  V10 J1 concordance identity re-derivation from raw records:
      max |flip_density - (1-tau)/2| over all 136 judge pairs ~= 3.0e-05
"""
import glob
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import binom, kendalltau

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLIT_CRIT = 0.5 + 0.10  # prereg v7 section 1.3(b): majority rate <= 0.6 = contested

ok, fail = [], []


def check(label, cond, detail=""):
    (ok if cond else fail).append(label)
    print(("  PASS " if cond else "  FAIL ") + label + ("  " + detail if detail else ""))


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


# ---------- V1 judge_raw integrity ----------
print("V1 judge_raw integrity")
files = sorted(glob.glob(str(DATA / "judge_raw" / "*.jsonl")))
n_rec = 0
battles = defaultdict(dict)  # bid -> judge -> direction ('A'/'B'/'C')
for fp in files:
    judge = Path(fp).stem
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line)
        assert all(k in r for k in ("id", "llm_judge", "winner_pred", "winner_gt",
                                    "prob_A", "prob_B", "prob_C")), "missing field"
        n_rec += 1
        battles[r["id"]][judge] = r["winner_pred"]
check("17 judge files", len(files) == 17, f"found {len(files)}")
check("848,946 judge-level records", n_rec == 848946, f"found {n_rec}")
check("848,946 = 49,938 battles x 17 judges", len(battles) * 17 == n_rec,
      f"{len(battles)} battles")

# ---------- V2 census re-derivation ----------
print("V2 census re-derivation (majority rate <= 0.6)")
contested = {}
for bid, prefs in battles.items():
    vals = [v for v in prefs.values() if v in ("A", "B")]  # drop tie votes
    if not vals:
        continue
    top = max(set(vals), key=vals.count)
    contested[bid] = 1 if vals.count(top) / len(vals) <= SPLIT_CRIT else 0
n_b = len(contested)
bad_b = int(sum(contested.values()))
share = bad_b / n_b
v7 = json.load(open(DATA / "v7_verdict.json", encoding="utf-8"))
check("n = 49,667 battles judged", n_b == 49667, f"recomputed {n_b}")
check("contested = 6,083", bad_b == 6083, f"recomputed {bad_b}")
check("contested share = 12.25%", close(share, v7["census"]["contested_share"]),
      f"recomputed {share:.4f} vs stored {v7['census']['contested_share']}")

# ---------- V3 Clopper-Pearson bounds ----------
print("V3 Clopper-Pearson one-sided 90% upper bounds")
cp = float(_beta.ppf(0.9, bad_b + 1, n_b - bad_b))
check("census CP bound 0.1244", close(cp, v7["census"]["cp_bound"]),
      f"recomputed {cp:.4f} vs stored {v7['census']['cp_bound']}")
v6 = json.load(open(DATA / "v6_verdict.json", encoding="utf-8"))
ac = v6["armA_census"]
cp6 = float(_beta.ppf(0.9, ac["bad"] + 1, ac["n"] - ac["bad"]))
check("v6 CP bound 0.1456 from 3/44", close(cp6, ac["bound"]),
      f"recomputed {cp6:.4f} vs stored {ac['bound']}")
check("v6 census 3/44, tier T2", ac["n"] == 44 and ac["bad"] == 3 and ac["tier"] == "T2")
sc = json.load(open(DATA / "v7_stratum_cert.json", encoding="utf-8"))
for tier, key in (("high", "high"), ("mid", "mid"), ("low", "low")):
    t = sc[key]
    cpt = float(_beta.ppf(0.9, t["bad"] + 1, t["n"] - t["bad"]))
    check(f"stratum {tier} CP bound", close(cpt, t["cp_bound"]),
          f"recomputed {cpt:.6f} vs stored {t['cp_bound']:.6f}")

# ---------- V4 cross-file consistency ----------
print("V4 cross-file consistency")
check("stratum overall == census", sc["overall"]["n"] == n_b and sc["overall"]["bad"] == bad_b)
cert_n = sc["high"]["n"] + sc["mid"]["n"]
check("certified tier high+mid = 34,269 (69.0%)", cert_n == 34269,
      f"{sc['high']['n']}+{sc['mid']['n']} = {cert_n}")
check("mid-tier CP bound <= 0.072", sc["mid"]["cp_bound"] <= 0.072,
      f"{sc['mid']['cp_bound']:.6f}")
rs = json.load(open(DATA / "v7_rank_stability.json", encoding="utf-8"))
check("rank-stability cert battles == certified tier", rs["battles"]["cert"] == cert_n)
boot = json.load(open(DATA / "v7_cluster_boot.json", encoding="utf-8"))
check("bootstrap low tier == stratum low", boot["tiers_block"]["low"]["n"] == sc["low"]["n"])
check("bootstrap census == v7 census",
      boot["census"]["n"] == n_b and close(boot["census"]["share"], share))

# ---------- V5 E2 exact McNemar ----------
print("V5 E2 exact McNemar")
e2 = json.load(open(DATA / "v7_e2e6e9.json", encoding="utf-8"))["E2"]["clause_i"]
p_exact = 2.0 * binom.cdf(0, e2["b"] + e2["c"], 0.5)
check("b=7, c=0", e2["b"] == 7 and e2["c"] == 0)
check("exact p = 0.0156 from 2*Bin(7,0.5)", close(p_exact, e2["exact_p"]),
      f"recomputed {p_exact:.4f} vs stored {e2['exact_p']}")

# ---------- V6 E3 retroactive probe ----------
print("V6 E3 retroactive probe")
e3 = json.load(open(DATA / "v7_e3_probe.json", encoding="utf-8"))
n_flip = len(e3["flips"])
n_persist = e3["persist_count"]
check("9 threshold flips", n_flip == 9, f"found {n_flip}")
check("2 persist -> 7 resolve", n_persist == 2 and n_flip - n_persist == 7)
check("x2|docunet excluded (v8)", "x2|docunet" not in [f["pair"] for f in e3["flips"]])

# ---------- V7 E5 stratification + Krippendorff alpha ----------
print("V7 E5 near-tie stratification + corrected K-alpha")
e5 = json.load(open(DATA / "v7_e5_kalpha.json", encoding="utf-8"))
sa = e5["e5_strat"]["scoring(A_vs_R)"]
check("scoring far-band pi = 0.0 (noise signature)", sa["pi_far75"] == 0.0)
check("gold far-band pi = 0.4947 (substantive)", close(e5["e5_strat"]["gold(gD_vs_gR)"]["pi_far75"], 0.4947))
check("corrected 3-class alpha 0.4561 present", close(e5["krippendorff_alpha_3class"], 0.4561))
check("corrected direction-only alpha 0.5035 present",
      close(e5["krippendorff_alpha_direction"], 0.5035))
check("stale first-pass alpha (-0.0878) absent",
      "krippendorff_alpha" not in e5 or not close(e5.get("krippendorff_alpha", 0), -0.0878, 1e-3))

# ---------- V8 E7 rank stability ----------
print("V8 Bradley-Terry rank stability")
check("tau(cert vs full) = 0.9977", close(rs["cert_vs_all"]["kendall_tau"], 0.9977))
check("cert + hedge = 49,938 = total battles",
      rs["battles"]["cert"] + rs["battles"]["hedge"] == len(battles))

# ---------- V9 NA-margin endpoint ----------
print("V9 NA-margin endpoint")
nm = json.load(open(DATA / "na_margin_flips.json", encoding="utf-8"))
flips = nm["flips"]
check("69 system pairs", len(flips) == 69, f"found {len(flips)}")
check("69/69 flip on the gold axis", all(f["flip"] for f in flips),
      f"{sum(1 for f in flips if f['flip'])}/69")

# ---------- V10 J1 concordance identity from raw records ----------
print("V10 J1 concordance identity re-derivation (136 judge pairs)")
pref = {j: {} for j in (Path(f).stem for f in files)}
for bid, prefs in battles.items():
    for j, v in prefs.items():
        if v in ("A", "B"):
            pref[j][bid] = 1 if v == "A" else 0
judges = sorted(pref)
dev = 0.0
for a in range(len(judges)):
    xa_all = pref[judges[a]]
    for b in range(a + 1, len(judges)):
        xb_all = pref[judges[b]]
        common = [bid for bid in xa_all if bid in xb_all]
        if len(common) < 1000:
            continue
        xa = np.array([xa_all[bid] for bid in common])
        xb = np.array([xb_all[bid] for bid in common])
        flip = float(np.mean(xa != xb))
        tau = float(kendalltau(xa, xb).statistic)
        dev = max(dev, abs(flip - (1 - tau) / 2))
check("max identity deviation ~= 3.0e-05", close(dev, v7["J1_identity_max_dev"], 5e-6),
      f"recomputed {dev:.3e} vs stored {v7['J1_identity_max_dev']:.3e}")

# ---------- summary ----------
print(f"\n===== {len(ok)} PASS / {len(fail)} FAIL =====")
for f_ in fail:
    print("  FAIL: " + f_)
sys.exit(1 if fail else 0)
