"""Figures for the report from results/summary.csv + per_utterance.csv. Palette = validated categorical slots 1-4 (light)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json, sys
sys.path.insert(0, 'src')
import metrics as M
from data import load as load_manifest

S1, S2, S3, S4, S5, S6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
OUT = Path("report/fig"); OUT.mkdir(parents=True, exist_ok=True)
s = pd.read_csv("results/summary.csv"); d = pd.read_csv("results/per_utterance.csv")
g = lambda sy, co, lg, m: float(s[(s.system == sy) & (s.cond == co) & (s.lang == lg)][m].iloc[0])
has = lambda sy, co="clean": ((s.system == sy) & (s.cond == co)).any()
MAN = {x["uid"]: x for x in load_manifest()}


def asr_wer(fname, lg):
    """WER straight from an ASR output file (independent of whether the LLM stages ran)."""
    rows = [json.loads(l) for l in open(f"results/{fname}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if MAN[r["uid"]]["lang"] == lg]
    return 100 * M.wer([MAN[r["uid"]]["ref_text"] for r in rows], [r["hyp_text"] for r in rows])


def clean_ax(ax):
    for k in ("top", "right", "left"): ax.spines[k].set_visible(False)
    ax.tick_params(length=0); ax.yaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)


def bars(ax, groups, series, vals, colors, fmt="{:.0f}", ylim=None):
    n = len(series); w = .8 / n
    for i, (nm, c) in enumerate(zip(series, colors)):
        xs = np.arange(len(groups)) + (i - (n - 1) / 2) * w
        ax.bar(xs, vals[i], width=w - .04, color=c, label=nm, zorder=3)
        for x, v in zip(xs, vals[i]):
            ax.text(x, v + (ylim or 100) * .015, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups); clean_ax(ax)
    if ylim: ax.set_ylim(0, ylim)


# Fig A: response correctness and hallucination by system x clean/noisy
SYS = [("base", "whisper-small + Qwen (baseline)"), ("base+mt", "whisper-small + NLLB + Qwen"),
       ("turbo", "large-v3-turbo + Qwen"), ("turbo+mt", "large-v3-turbo + NLLB + Qwen"),
       ("conf", "IndicConformer + Qwen"), ("conf+mt", "IndicConformer + NLLB + Qwen")]
SYS = [x for x in SYS if has(x[0])]; COL = [S1, S2, S3, S4, S5, S6][:len(SYS)]
conds = [c for c, _ in (("clean", "clean"), ("noisy", "5 dB noise")) if all(has(sy, c) for sy, _ in SYS)] or ["clean"]
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)
for ax, m, t in zip(axs, ["correct", "halluc"], ["Correct responses (%)  higher is better", "Hallucinated responses (%)  lower is better"]):
    bars(ax, [{"clean": "clean", "noisy": "5 dB noise"}[c] for c in conds], [n for _, n in SYS],
         [[100 * g(sy, c, "all", m) for c in conds] for sy, _ in SYS], COL, ylim=105)
    ax.set_title(t, fontsize=9, loc="left", color=INK)
axs[0].legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0, -.12), ncol=2)
fig.tight_layout(); fig.savefig(OUT / "A_systems.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Fig B: WER by language x condition: whisper-small, turbo, IndicConformer (RNNT)
SRC = [("whisper-small", "{c}", "base"), ("whisper-large-v3-turbo", "asr-turbo_{c}", None), ("IndicConformer (RNNT)", "asr-conf-rnnt_{c}", None)]
SRC = [("whisper-small", "base_{c}"), ("whisper-large-v3-turbo", "asr-turbo_{c}"), ("IndicConformer (RNNT)", "asr-conf-rnnt_{c}")]
SRC = [x for x in SRC if all(Path(f"results/{x[1].format(c=c)}.jsonl").exists() for c in ("clean", "noisy"))]
fig, ax = plt.subplots(figsize=(6.2, 3.1))
grp = [("hi", "clean"), ("ta", "clean"), ("hi", "noisy"), ("ta", "noisy")]
bars(ax, ["Hindi clean", "Tamil clean", "Hindi 5 dB", "Tamil 5 dB"], [n for n, _ in SRC],
     [[asr_wer(f.format(c=c), l) for l, c in grp] for _, f in SRC], [S1, S3, S5][:len(SRC)], ylim=115)
ax.set_title("ASR word error rate (%)  lower is better", fontsize=9, loc="left", color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "B_wer.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Fig C: per-language decomposition (clean): transcript source x response stage -> correct %
have_mt = (s.system == "mtonly-base").any()
if have_mt:
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    stages = ["LLM only", "NLLB only (no LLM)", "NLLB + LLM"]
    src = [("Gold transcript", S1, ["oracle", "mtonly-oracle", "oracle+mt"], "oracle"),
           ("whisper-small", S2, ["base", "mtonly-base", "base+mt"], "clean"),
           ("large-v3-turbo", S3, ["turbo", "mtonly-turbo", "turbo+mt"], "clean"),
           ("IndicConformer", S4, ["conf", "mtonly-conf", "conf+mt"], "clean")]
    src = [x for x in src if all(has(sy, x[3]) for sy in x[2])]
    for ax, (lg, name) in zip(axs, [("hi", "Hindi"), ("ta", "Tamil")]):
        bars(ax, stages, [x[0] for x in src], [[100 * g(sy, x[3], lg, "correct") for sy in x[2]] for x in src],
             [x[1] for x in src], ylim=110)
        ax.set_title(f"{name}: correct responses (%), clean audio", fontsize=9, loc="left", color=INK)
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "C_decomp.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Fig D: response quality vs utterance WER (baseline + turbo pooled)
d2 = d[d.system.isin(["base", "turbo", "conf"])].copy()
d2["wb"] = pd.cut(d2.utt_wer, [-.01, .2, .4, .6, .8, 10], labels=["≤20", "20–40", "40–60", "60–80", ">80"])
t = d2.groupby("wb", observed=True).agg(n=("cos", "size"), correct=("correct", "mean"), halluc=("halluc", "mean"))
fig, ax = plt.subplots(figsize=(5.6, 3.0))
bars(ax, [f"{i}\n(n={int(n)})" for i, n in zip(t.index, t.n)], ["correct", "hallucinated"],
     [list(100 * t.correct), list(100 * t.halluc)], [S1, S2], ylim=115)
ax.set_xlabel("Per-utterance ASR WER bucket (%) — whisper-small, turbo, IndicConformer; clean and noisy pooled")
ax.set_title("Response quality (%) falls as ASR errors rise", fontsize=9, loc="left", color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
fig.tight_layout(); fig.savefig(OUT / "D_wer_vs_quality.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
