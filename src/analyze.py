"""Aggregate results/*.jsonl -> results/summary.csv, per-utterance csv, figures.
usage: python src/analyze.py base improved   (tags; 'oracle' is added automatically if present)"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
from data import load
import metrics as M

items = {x["uid"]: x for x in load()}
tags = sys.argv[1:] or ["base"]
rows = []
for f in sorted(glob.glob("results/*.jsonl")):
    stem = Path(f).stem
    if stem.startswith("gold") or (not stem.startswith("oracle") and stem.rsplit("_", 1)[0] not in tags):
        continue
    lines = [l for l in open(f, encoding="utf-8") if l.strip()]
    if len(lines) != len(items):  # unfinished stage: never score a partial file
        print(f"skipping incomplete {f} ({len(lines)}/{len(items)})"); continue
    for l in lines:
        r = json.loads(l); it = items[r["uid"]]
        rows.append({**r, "system": stem.rsplit("_", 1)[0], "lang": it["lang"],
                     "ref": it["ref_text"], "ref_en": it["ref_en"], "dur": it["duration"]})
df = pd.DataFrame(rows)
df["utt_wer"] = [M.utt_wer(r, h) for r, h in zip(df.ref, df.hyp_text)]
E = M.embed(list(df.response) + list(df.ref_en))
n = len(df); cos = (E[:n] * E[n:]).sum(1); df["cos"] = cos
# Thresholds were set by manually inspecting oracle (gold-transcript) responses across the cosine range:
# <0.4 unrelated/fabricated, ~0.45 half wrong, 0.6 mostly faithful, >=0.73 faithful. Sensitivity is reported in the README/report.
tau_c = 0.6
df["correct"] = df.cos >= tau_c
for t in (0.5, 0.7):
    df[f"correct_{t}"] = df.cos >= t
fl = [M.hallucination_flags(r, e, c) for r, e, c in zip(df.response, df.ref_en, df.cos)]
df["h_num"], df["h_drift"], df["h_verbose"] = zip(*fl)
df["halluc"] = df[["h_num", "h_drift", "h_verbose"]].any(axis=1)
df["chrf"] = [M.chrf([h], [r]) for h, r in zip(df.response, df.ref_en)]
df["t_total"] = df.get("t_asr", 0).fillna(0) + df.t_llm
df.to_csv("results/per_utterance.csv", index=False)

def agg(g):
    has_asr = g.t_asr.notna().all()  # oracle rows have no ASR stage
    d = dict(n=len(g), chrF=M.chrf(list(g.response), list(g.ref_en)), cos=g.cos.mean(), correct=g.correct.mean(),
             halluc=g.halluc.mean(), h_num=g.h_num.mean(), h_drift=g.h_drift.mean(), h_verbose=g.h_verbose.mean(),
             llm_s=g.t_llm.mean())
    if has_asr:
        lo, hi = M.bootstrap_ci(list(g.ref), list(g.hyp_text), M.wer)
        d.update(WER=M.wer(g.ref, g.hyp_text) * 100, WER_lo=lo * 100, WER_hi=hi * 100,
                 CER=M.cer(g.ref, g.hyp_text) * 100, asr_RTF=g.t_asr.sum() / g.dur.sum(),
                 total_RTF=g.t_total.sum() / g.dur.sum(), asr_s=g.t_asr.mean(), total_s=g.t_total.mean())
    return pd.Series(d)

recs = []
for (sy, co), g in df.groupby(["system", "cond"]):
    for lg, gg in [("all", g)] + [(l, x) for l, x in g.groupby("lang")]:
        recs.append(dict(system=sy, cond=co, lang=lg, **agg(gg).to_dict()))
summ = pd.DataFrame(recs)
summ.to_csv("results/summary.csv", index=False)
json.dump(dict(tau_correct=tau_c), open("results/thresholds.json", "w"))
pd.set_option("display.width", 250, "display.max_columns", 30)
print("tau_correct =", round(tau_c, 3)); print(summ.round(3).to_string(index=False))
