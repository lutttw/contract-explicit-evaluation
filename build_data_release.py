# -*- coding: utf-8 -*-
"""build_data_release.py — 按 Data availability 承诺配齐仓库:
frozen per-document counts / contract readings / learned weights /
evaluation harnesses / per-instance artifacts。"""
import io
import re
import shutil
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REL = Path(__file__).resolve().parent
D = Path(r"E:\WorkBuddy\参考文献解读\04_文档级IE")
SAGG = REL.parent                                   # paper3_stable_agg

copied, skipped = [], []


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst.relative_to(REL)))


def cp_pattern(base, pattern, dstsub):
    if not base.exists():
        skipped.append(f"{base} (missing)")
        return
    for f in sorted(base.rglob(pattern)):
        if not f.is_file():
            continue
        cp(f, REL / "data" / dstsub / f.relative_to(base))


# 1 frozen per-document counts — X2 audit runs (04-docie side)
cp_pattern(D / "out" / "jcip_audit_v1" / "runs", "evaluation__*.json", "perdoc_counts/audit_runs")
# 1b growth runs (paper3_stable_agg side; mirrors referenced by v6 census)
cp_pattern(SAGG / "out" / "x2_v4" / "runs", "evaluation__*.json", "perdoc_counts/growth_runs")
# 1c P3M bridged per_doc
cp_pattern(D / "out" / "jcip_audit_v1" / "x3_bridged", "per_doc_*", "perdoc_counts/x3_bridged")
# 1d published-family per_doc json
cp_pattern(D / "out" / "evallever_release_v1_submission", "per_doc_*.json", "perdoc_counts/families")

# 2 contract readings registry + per-instance conformal record
cp(SAGG / "out" / "confirmatory_v3" / "registry_v3.json", REL / "data" / "registry_v3.json")
cp(SAGG / "out" / "confirmatory_v3" / "confirmatory_result.json",
   REL / "data" / "confirmatory_result.json")

# 3 learned weights + feature/count matrices
for n in ("agg_v02_weights.npz", "features.npz", "llm_counts.npz"):
    cp(SAGG / "out" / n, REL / "data" / "learned_weights" / n)

# 4 golds + id lists (reading-space golds consumed by the loader)
prep = D / "out" / "lre_rebuttal_v2" / "prepared"
for n in ("golds_gD.jsonl", "golds_gR.jsonl", "golds_gR2.jsonl",
          "relation_to_index.json", "calibration_ids.json", "evaluation_ids.json"):
    if (prep / n).exists():
        cp(prep / n, REL / "data" / "golds_prepared" / n)
    else:
        skipped.append(f"prepared/{n} (missing)")

# 5 per-instance candidate artifacts (train/calibration/evaluation)
cand = prep / "candidates"
for n in ("train.jsonl", "calibration.jsonl", "evaluation.jsonl"):
    f = cand / n
    if f.exists():
        mb = f.stat().st_size / 1e6
        if mb < 80:
            cp(f, REL / "data" / "candidates" / n)
        else:
            skipped.append(f"candidates/{n} ({mb:.0f}MB > 80MB cap)")
    else:
        skipped.append(f"candidates/{n} (missing)")

# 6 evaluation harness (path-scrubbed)
for n in ("confirmatory_run.py", "p3loader.py", "p3io.py"):
    t = (SAGG / "src" / n).read_text(encoding="utf-8")
    t2 = t.replace("E:" + chr(92) + "WorkBuddy" + chr(92) + "参考文献解读"
                   + chr(92) + "04_文档级IE", "<local-workspace>")
    t2 = t2.replace("E:" + chr(92) + "WorkBuddy" + chr(92) + "参考文献解读"
                    + chr(92) + "paper3_stable_agg", "<local-workspace>")
    note = ("# released copy: as-run internal paths replaced by <local-workspace>;\n"
            "# point the ROOT constant at a directory laid out like the release data/.\n")
    (REL / "code" / n).write_text(note + t2, encoding="utf-8", newline="\n")
    copied.append(f"code/{n}")

tot = sum(f.stat().st_size for f in REL.rglob("*") if f.is_file() and ".git" not in str(f))
print(f"copied {len(copied)} files; skipped: {skipped}")
print(f"repo total now: {tot/1e6:.0f} MB")
big = [(str(f.relative_to(REL)), f.stat().st_size // 1e6)
       for f in REL.rglob("*") if f.is_file() and f.stat().st_size > 50e6 and ".git" not in str(f)]
print("files >50MB:", big or "none")
