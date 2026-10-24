"""Aggregate the Belebele-Fleurs listening-comprehension results -> results/qa_summary.csv, results/qa_buckets.csv, report/fig/E_qa.png.
Accuracy is exact (chosen option == gold option). Chance = 25%. 95% CIs: Wilson (single accuracy) / paired bootstrap (differences)."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import belebele, metrics as M

TAGS = [("gold", "Gold transcript"), ("small", "whisper-small"), ("turbo", "whisper-large-v3-turbo"), ("conf", "IndicConformer")]
MODES = [("native", "native-language prompt"), ("en", "NLLB-translated to English")]
qs = belebele.load(); Q = {q["qid"]: q for q in qs}
R = Path("results/qa")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


rows, per = [], []
for tag, name in TAGS:
    asr = {r["sid"]: r for r in map(json.loads, open(R / f"asr_{tag}.jsonl", encoding="utf-8"))}
    for lg in ("hi", "ta"):
        sids = [s["sid"] for q in qs if q["lang"] == lg for s in q["sentences"]]; sids = list(dict.fromkeys(sids))
        refs = [next(s["ref"] for q in qs for s in q["sentences"] if s["sid"] == i) for i in sids]
        hyps = [asr[i]["hyp"] for i in sids]
        wer = M.wer(refs, hyps) * 100 if tag != "gold" else 0.0; cer = M.cer(refs, hyps) * 100 if tag != "gold" else 0.0
        rtf = sum(asr[i]["t"] for i in sids) / sum(asr[i]["dur"] for i in sids) if tag != "gold" else float("nan")
        for mode, mname in MODES:
            a = [json.loads(l) for l in open(R / f"ans_{tag}_{mode}.jsonl", encoding="utf-8")]; a = [x for x in a if x["lang"] == lg]
            k = sum(x["correct"] for x in a); n = len(a); lo, hi = wilson(k, n)
            conf_wrong = sum((not x["correct"]) and max(x["probs"]) >= 0.7 for x in a) / n
            rows.append(dict(source=tag, mode=mode, lang=lg, n=n, acc=100 * k / n, lo=100 * lo, hi=100 * hi, confident_wrong=100 * conf_wrong,
                             passage_WER=wer, passage_CER=cer, asr_RTF=rtf))
            for x in a:
                ids = [s["sid"] for s in Q[x["qid"]]["sentences"]]
                per.append(dict(source=tag, mode=mode, lang=lg, qid=x["qid"], correct=x["correct"],
                                wer=M.wer([next(s["ref"] for s in Q[x["qid"]]["sentences"] if s["sid"] == i) for i in ids], [asr[i]["hyp"] for i in ids]) if tag != "gold" else 0.0))
df = pd.DataFrame(rows)
# both languages pooled
for (tag, mode), g in df.groupby(["source", "mode"]):
    k = sum(g.acc * g.n / 100); n = g.n.sum(); lo, hi = wilson(k, n)
    rows.append(dict(source=tag, mode=mode, lang="all", n=n, acc=100 * k / n, lo=100 * lo, hi=100 * hi,
                     confident_wrong=float((g.confident_wrong * g.n).sum() / n), passage_WER=float("nan"), passage_CER=float("nan"), asr_RTF=float("nan")))
df = pd.DataFrame(rows); df.to_csv("results/qa_summary.csv", index=False)
pq = pd.DataFrame(per); pq.to_csv("results/qa_per_question.csv", index=False)

# accuracy by passage WER bucket (ASR sources only, pooled over modes/languages)
x = pq[pq.source != "gold"].copy(); x["bucket"] = pd.cut(x.wer, [-.01, .15, .3, .5, 10], labels=["≤15", "15–30", "30–50", ">50"])
b = x.groupby("bucket", observed=True).agg(n=("correct", "size"), acc=("correct", "mean")).reset_index(); b["acc"] *= 100
b.to_csv("results/qa_buckets.csv", index=False)

# paired bootstrap for key contrasts
def diff(a_src, a_mode, b_src, b_mode, lg=None, n=4000, seed=0):
    A = pq[(pq.source == a_src) & (pq["mode"] == a_mode)].set_index("qid").correct; B = pq[(pq.source == b_src) & (pq["mode"] == b_mode)].set_index("qid").correct
    if lg: A, B = A[A.index.str.startswith(lg)], B[B.index.str.startswith(lg)]
    d = (A - B.reindex(A.index)).values; r = np.random.default_rng(seed)
    bs = [r.choice(d, len(d)).mean() for _ in range(n)]
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
con = []
for lab, args in [("Conformer vs whisper-small (English prompt)", ("conf", "en", "small", "en")),
                  ("Conformer vs whisper-small (native prompt)", ("conf", "native", "small", "native")),
                  ("Turbo vs whisper-small (English prompt)", ("turbo", "en", "small", "en")),
                  ("English prompt vs native prompt (Conformer)", ("conf", "en", "conf", "native")),
                  ("English prompt vs native prompt (gold text)", ("gold", "en", "gold", "native")),
                  ("Gold vs Conformer (English prompt)", ("gold", "en", "conf", "en"))]:
    for lg in (None, "hi", "ta"):
        m, lo, hi = diff(*args, lg=lg); con.append(dict(contrast=lab, lang=lg or "all", diff_pts=m, lo=lo, hi=hi))
pd.DataFrame(con).to_csv("results/qa_contrasts.csv", index=False)

# figure
S1, S2, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
for ax, (lg, ln) in zip(axs, [("hi", "Hindi"), ("ta", "Tamil")]):
    for j, ((mode, mname), c) in enumerate(zip(MODES, [S1, S2])):
        v = [float(df[(df.source == t) & (df["mode"] == mode) & (df.lang == lg)].acc.iloc[0]) for t, _ in TAGS]
        e = [[v[i] - float(df[(df.source == t) & (df["mode"] == mode) & (df.lang == lg)].lo.iloc[0]) for i, (t, _) in enumerate(TAGS)],
             [float(df[(df.source == t) & (df["mode"] == mode) & (df.lang == lg)].hi.iloc[0]) - v[i] for i, (t, _) in enumerate(TAGS)]]
        xs = np.arange(len(TAGS)) + (j - .5) * .38
        ax.bar(xs, v, .34, color=c, label=mname, zorder=3); ax.errorbar(xs, v, yerr=e, fmt="none", ecolor=INK2, elinewidth=.8, capsize=2, zorder=4)
        for xx, vv, hh in zip(xs, v, e[1]): ax.text(xx, vv + hh + 1.5, f"{vv:.0f}", ha="center", fontsize=7.5)
    ax.axhline(25, color=INK2, lw=.9, ls=(0, (4, 3)), zorder=2, label="chance (25%)")
    ax.set_xticks(range(len(TAGS))); ax.set_xticklabels(["gold\ntext", "whisper-\nsmall", "whisper-\nturbo", "Indic-\nConformer"], fontsize=8)
    ax.set_title(f"{ln}: answer accuracy (%), 100 questions", fontsize=9, loc="left")
    for k in ("top", "right", "left"): ax.spines[k].set_visible(False)
    ax.tick_params(length=0); ax.yaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_ylim(0, 105)
axs[0].legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -.2), ncol=2)
fig.tight_layout(); Path("report/fig").mkdir(parents=True, exist_ok=True); fig.savefig("report/fig/E_qa.png", dpi=200, bbox_inches="tight")
pd.set_option("display.width", 220)
print(df[df.lang != "all"].round(1).to_string(index=False)); print(df[df.lang == "all"][["source", "mode", "n", "acc", "lo", "hi"]].round(1).to_string(index=False))
print(b.round(1).to_string(index=False)); print(pd.DataFrame(con).round(1).to_string(index=False))
