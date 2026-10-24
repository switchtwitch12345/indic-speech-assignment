"""Assemble report/report.html from results + task1.html, then (separately) print to PDF with headless Chrome."""
import base64, html, json
from pathlib import Path
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "report"); sys.path.insert(0, "src")
import figs

s = pd.read_csv("results/summary.csv"); d = pd.read_csv("results/per_utterance.csv")
lat = json.load(open("results/latency_bench.json"))
G = lambda sy, co, lg, m: float(s[(s.system == sy) & (s.cond == co) & (s.lang == lg)][m].iloc[0])
img = lambda n, w="100%": f'<img src="data:image/png;base64,{base64.b64encode(open("report/fig/"+n,"rb").read()).decode()}" style="width:{w}">'


def ci(sy, co, lg, col, n=2000, seed=0):
    x = d[(d.system == sy) & (d.cond == co)]
    x = x if lg == "all" else x[x.lang == lg]
    v = x[col].astype(float).values; rng = np.random.default_rng(seed)
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return 100 * v.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def pc(sy, co, lg, col):
    m, lo, hi = ci(sy, co, lg, col); return f"{m:.0f} <span class=ci>[{lo:.0f}–{hi:.0f}]</span>"


def _has(sy, co="clean"): return bool(((s.system == sy) & (s.cond == co)).any())
SYS = [("base", "A0  whisper-small → Qwen2.5-3B (baseline)"), ("base+mt", "B   whisper-small → NLLB → Qwen"),
       ("mtonly-base", "B′  whisper-small → NLLB only (control, no LLM)"), ("turbo", "A   large-v3-turbo → Qwen"),
       ("turbo+mt", "A+B large-v3-turbo → NLLB → Qwen"), ("mtonly-turbo", "A+B′ large-v3-turbo → NLLB only (control)"),
       ("conf", "C   IndicConformer → Qwen"), ("conf+mt", "C+B IndicConformer → NLLB → Qwen"),
       ("mtonly-conf", "C+B′ IndicConformer → NLLB only (control)")]
SYS = [x for x in SYS if _has(x[0])]


def table_main(cond):
    r = "".join(f"<tr><td>{n}</td><td>{G(sy,cond,'all','WER'):.1f}</td><td>{G(sy,cond,'all','CER'):.1f}</td>"
                f"<td>{pc(sy,cond,'all','correct')}</td><td>{pc(sy,cond,'all','halluc')}</td>"
                f"<td>{G(sy,cond,'all','chrF'):.1f}</td></tr>" for sy, n in SYS if _has(sy, cond))
    return ("<table><tr><th>System</th><th>WER %</th><th>CER %</th><th>Correct % [95% CI]</th><th>Hallucinated % [95% CI]</th>"
            f"<th>chrF</th></tr>{r}</table>")


def table_lang():
    rows = ""
    for sy, n in [x for x in SYS if x[0] in ("base", "turbo", "conf")]:
        for co in ("clean", "noisy"):
            if not _has(sy, co): continue
            for lg, ln in (("hi", "Hindi"), ("ta", "Tamil")):
                rows += (f"<tr><td>{n.split('  ',1)[1] if '  ' in n else n}</td><td>{co}</td><td>{ln}</td><td>{G(sy,co,lg,'WER'):.1f}</td>"
                         f"<td>{G(sy,co,lg,'CER'):.1f}</td><td>{100*G(sy,co,lg,'correct'):.0f}</td><td>{100*G(sy,co,lg,'halluc'):.0f}</td></tr>")
    return "<table><tr><th>ASR + LLM</th><th>Audio</th><th>Language</th><th>WER %</th><th>CER %</th><th>Correct %</th><th>Halluc. %</th></tr>" + rows + "</table>"


def table_oracle():
    r = ""
    for sy, n in [("oracle", "Gold transcript → Qwen"), ("mtonly-oracle", "Gold transcript → NLLB only"), ("oracle+mt", "Gold transcript → NLLB → Qwen")]:
        r += f"<tr><td>{n}</td>" + "".join(f"<td>{100*G(sy,'oracle',l,'correct'):.0f}</td><td>{100*G(sy,'oracle',l,'halluc'):.0f}</td>" for l in ("hi", "ta", "all")) + "</tr>"
    return "<table><tr><th rowspan=2>Input to response stage</th><th colspan=2>Hindi</th><th colspan=2>Tamil</th><th colspan=2>All</th></tr><tr>" + "<th>Correct %</th><th>Halluc. %</th>" * 3 + f"</tr>{r}</table>"


def _asr_rows():
    man = {x["uid"]: x for x in map(json.loads, open("data/manifest.jsonl", encoding="utf-8"))}
    import metrics as M2
    out = ""
    for name, f in [("whisper-small (greedy)", "base_{c}"), ("whisper-large-v3-turbo (greedy)", "asr-turbo_{c}"),
                    ("IndicConformer-600M, CTC decoding", "asr-conf-ctc_{c}"), ("IndicConformer-600M, RNNT decoding", "asr-conf-rnnt_{c}")]:
        cells = ""
        for c in ("clean", "noisy"):
            pth = Path("results/" + f.format(c=c) + ".jsonl")
            if not pth.exists(): cells += "<td>–</td><td>–</td><td>–</td><td>–</td>"; continue
            rs = [json.loads(l) for l in open(pth, encoding="utf-8")]
            for lg in ("hi", "ta"):
                x = [r for r in rs if man[r["uid"]]["lang"] == lg]
                refs, hyps = [man[r["uid"]]["ref_text"] for r in x], [r["hyp_text"] for r in x]
                cells += f"<td>{100*M2.wer(refs,hyps):.1f} / {100*M2.cer(refs,hyps):.1f}</td>"
        out += f"<tr><td>{name}</td>{cells}</tr>"
    return ("<table><tr><th rowspan=2>ASR system</th><th colspan=2>Clean: WER / CER %</th><th colspan=2>5 dB noise: WER / CER %</th></tr>"
            "<tr><th>Hindi</th><th>Tamil</th><th>Hindi</th><th>Tamil</th></tr>" + out + "</table>")


st = lat["stages"]; A = lat["audio_s"]; N = lat["n"]
nl = st["nllb"]["total_s"] / 2  # NLLB timed over two hypothesis sets
PIPE = [("whisper-small → Qwen (baseline)", st["whisper-small"]["total_s"] + st["qwen_plain"]["total_s"]),
        ("large-v3-turbo → Qwen", st["whisper-large-v3-turbo"]["total_s"] + st["qwen_plain"]["total_s"]),
        ("large-v3-turbo → NLLB only", st["whisper-large-v3-turbo"]["total_s"] + nl),
        ("large-v3-turbo → NLLB → Qwen", st["whisper-large-v3-turbo"]["total_s"] + nl + st["qwen_grounded"]["total_s"])]
lat_rows = "".join(f"<tr><td>{n}</td><td>{t/N:.2f}</td><td>{t/A:.2f}</td></tr>" for n, t in PIPE)
NAMES = {"whisper-small": "ASR: whisper-small", "whisper-large-v3-turbo": "ASR: whisper-large-v3-turbo", "nllb": "NLLB-600M translation", "qwen_plain": "Qwen2.5-3B, transcript only", "qwen_grounded": "Qwen2.5-3B, transcript + NLLB draft"}
stage_rows = "".join(f"<tr><td>{NAMES[k]}</td><td>{(v['total_s']/(2 if k=='nllb' else 1))/N:.2f}</td></tr>" for k, v in st.items())

# qualitative examples: first two utterances of each language in manifest order (not cherry-picked)
man = [json.loads(l) for l in open("data/manifest.jsonl", encoding="utf-8")]
ex_uids = [m["uid"] for lg in ("hi", "ta") for m in [x for x in man if x["lang"] == lg][:2]]
ex = ""
for u in ex_uids:
    m = next(x for x in man if x["uid"] == u); row = lambda sy: d[(d.system == sy) & (d.cond == ("oracle" if sy.startswith("oracle") else "clean")) & (d.uid == u)].iloc[0]
    b, t = row("base"), row("turbo+mt")
    ex += (f"<div class=ex><b>{u}</b> &middot; reference (English): {html.escape(m['ref_en'])}<br>"
           f"<i>Reference transcript:</i> {html.escape(m['ref_text'])}<br>"
           f"<i>Baseline ASR:</i> {html.escape(str(b.hyp_text))}<br><i>Baseline response</i> (cos {b.cos:.2f}): {html.escape(str(b.response))}<br>"
           f"<i>Turbo ASR:</i> {html.escape(str(t.hyp_text))}<br><i>Turbo+NLLB+Qwen response</i> (cos {t.cos:.2f}): {html.escape(str(t.response))}</div>")

t1 = open("report/task1.html", encoding="utf-8").read().replace("%%MOSHI%%", figs.moshi()).replace("%%QWEN%%", figs.qwen()).replace("%%PROPOSAL%%", figs.proposal())
b0, bn = G("base", "clean", "all", "WER"), G("base", "noisy", "all", "WER")
CSS = """
@page{size:A4;margin:16mm 15mm}
body{font:10pt/1.45 Helvetica,Arial,sans-serif;color:#111;max-width:none;margin:0;padding:0}
h1{font-size:19pt;margin:0 0 2pt}h2{font-size:14pt;border-bottom:1.5px solid #333;padding-bottom:2pt;margin-top:20pt;page-break-after:avoid}
h3{font-size:11pt;margin:13pt 0 3pt;page-break-after:avoid}table{border-collapse:collapse;width:100%;margin:6pt 0;font-size:8.6pt}tr{page-break-inside:avoid}.cmp{page-break-inside:auto}
th,td{border:1px solid #bbb;padding:3pt 5pt;vertical-align:top;text-align:left}th{background:#eef0f3}.cmp td:first-child{font-weight:600;width:13%}
figure{margin:8pt 0;page-break-inside:avoid}figcaption{font-size:8.5pt;color:#444}.note{background:#f6f7f9;border-left:3px solid #888;padding:5pt 8pt;font-size:9pt}
.ci{color:#666;font-size:7.5pt}.ex{border:1px solid #ccc;padding:5pt 7pt;margin:5pt 0;font-size:8.3pt;page-break-inside:avoid}.warn{background:#fff6e5;border-left:3px solid #d98a00;padding:5pt 8pt;font-size:9pt}
img{max-width:100%}svg{max-width:100%;height:auto}small{color:#555}
"""
_d2 = d[d.system.isin(["base", "turbo", "conf"])].copy()
_d2["wb"] = pd.cut(_d2.utt_wer, [-.01, .2, .4, .6, .8, 10], labels=["a", "b", "c", "d", "e"])
_b = _d2.groupby("wb", observed=True).agg(correct=("correct", "mean"), halluc=("halluc", "mean"))
T2 = f"""
<h2>Task 2 &mdash; Indic speech → text-response system and evaluation</h2>
<p class=note><b>What was built.</b> A reproducible pipeline: speech → ASR → LLM → English text response, evaluated on 120 FLEURS test utterances
(60 Hindi, 60 Tamil; 23.7 min of audio) under two acoustic conditions (clean; white noise at 5 dB SNR), with my own evaluation code
(WER/CER, response correctness, hallucination, latency). All models are open-source and run locally on one Apple M2 Pro (16 GB, MPS). The &ldquo;response&rdquo; task is
<i>say in English what the speaker said</i>: the only task for which a reference answer exists for every clip (FLEURS clips are aligned to FLORES English text), which makes correctness and hallucination measurable without a human judge.</p>

<h3>2.1 Setup and metric definitions</h3>
<ul>
<li><b>Data.</b> FLEURS test split, Hindi and Tamil, one clip per sentence id drawn with a fixed seed (0). English reference = the FLORES-aligned English sentence for the same id. <b>Noisy condition</b> = clean audio + white Gaussian noise at 5 dB SNR, seeded per utterance.</li>
<li><b>Baseline models.</b> ASR: openai/whisper-small (language forced to the known language, greedy). LLM: Qwen2.5-3B-Instruct (bf16, greedy), prompted with the transcript to reply with a faithful English translation and not to add information.</li>
<li><b>ASR metrics.</b> WER and CER against the FLEURS raw transcript, both sides NFC-normalised, lower-cased, punctuation stripped (incl. danda). Corpus-level; 95% CI by bootstrap over utterances.</li>
<li><b>Response correctness.</b> Cosine similarity (all-MiniLM-L6-v2) between the response and the English reference; <b>correct</b> if cos ≥ 0.6. I also report chrF.</li>
<li><b>Hallucination.</b> A response is flagged if any of: (H1) it states a number absent from the reference; (H2) semantic drift, cos &lt; 0.4 (content unrelated to what was said); (H3) it is more than 2× longer than the reference + 5 words (content added beyond a translation).</li>
<li><b>Latency.</b> Wall-clock per stage with MPS synchronisation, batch size 1; real-time factor RTF = processing time ÷ audio duration (RTF &lt; 1 is faster than real time). Measured in a dedicated benchmark (see 2.5).</li>
</ul>
<div class=warn><b>Honest limits of the response metrics.</b> Correctness and hallucination are <i>automatic proxies</i>, not human judgements. The 0.6 / 0.4 thresholds were set by me after reading gold-transcript outputs across the cosine range (&lt;0.4 unrelated, ≈0.45 half wrong, 0.6 mostly faithful, ≥0.73 faithful); they were not calibrated against human labels. Treat absolute percentages as indicative and the <i>comparisons between systems</i> as the result. A human-labelled audit of ~100 responses is the obvious next step. The embedding model also scores Tamil/Hindi-flavoured transliterations of names loosely, which depresses scores somewhat equally for all systems.</div>

<h3>2.2 Baseline results</h3>
<p>Clean audio:</p>{table_main("clean")}
<p>5 dB noise:</p>{table_main("noisy")}
<p><small>Rows B′ and A+B′ are controls explained in 2.4. WER/CER do not depend on the LLM stage, so rows sharing an ASR share WER/CER.</small></p>
<h3>Performance across conditions</h3>{table_lang()}
<p>Baseline whisper-small is weak on Indic speech: {b0:.0f}% WER on clean audio and {bn:.0f}% at 5 dB, i.e. nearly every word is wrong under noise. Tamil has a much lower CER than WER (23% vs 73%) because it is agglutinative: one wrong suffix makes the whole word wrong, so WER overstates how much of the text is wrong. Hindi is hurt more by noise (CER 33% → 70%).</p>

<h3>2.3 Bottleneck analysis</h3>
<figure>{img("D_wer_vs_quality.png","62%")}<figcaption>Fig. 4. Response quality against per-utterance ASR error (whisper-small, turbo and IndicConformer; clean and noisy pooled). Correctness falls from {_b.correct.iloc[0]*100:.0f}% (WER ≤ 20%) to {_b.correct.iloc[-1]*100:.0f}% (WER &gt; 80%); hallucination rises from {_b.halluc.iloc[0]*100:.0f}% to {_b.halluc.iloc[-1]*100:.0f}%.</figcaption></figure>
<p>ASR errors are the first measurable bottleneck: per-utterance WER and response similarity are negatively correlated (Spearman −0.57 for Hindi, −0.28 for Tamil in the baseline), and the LLM turns recognition errors into <i>fluent, confident, wrong</i> English (e.g. a mis-heard Hindi word turned &ldquo;local company&rdquo; into &ldquo;data company&rdquo;). But a second experiment shows ASR is not the whole story. Feeding <b>gold transcripts</b> to the same LLM removes ASR error entirely:</p>
{table_oracle()}
<p>With perfect Hindi text Qwen2.5-3B is mostly adequate (78% correct), so for Hindi the bottleneck is ASR. With perfect <b>Tamil</b> text it gets only 17% correct and hallucinates 65%: for Tamil the bottleneck is the LLM's comprehension. This is why upgrading the ASR helps Hindi a lot (WER 67 → 33%, correct 17 → 48%) but does nothing for Tamil responses (2% → 2%), even though Tamil WER also improves slightly (73 → 68%).</p>

<h3>2.4 Research extension: implementation and ablation</h3>
<p><b>Bottleneck (measured on the whisper-small baseline, see 2.3):</b> ASR for Hindi, and the 3B LLM&rsquo;s weak Tamil comprehension for Tamil. <b>Improvements tested</b> (a factorial plus controls, all on the same 120 clips):</p>
<ul><li><b>A &mdash; stronger ASR:</b> whisper-large-v3-turbo instead of whisper-small.</li>
<li><b>C &mdash; Indic-specialised ASR:</b> AI4Bharat IndicConformer-600M (hybrid CTC/RNNT Conformer trained on Indian languages; MIT licence, gated download), RNNT decoding chosen before looking at results; runs on CPU through ONNX.</li>
<li><b>B &mdash; translation grounding:</b> a dedicated Indic→English MT model (NLLB-200-distilled-600M) translates the transcript; the LLM receives the transcript <i>and</i> the draft translation and is told to use the draft as a guide.</li>
<li><b>B′ &mdash; control (no LLM):</b> the NLLB translation is returned directly. This tests whether the LLM adds anything beyond MT.</li></ul>
<figure>{img("A_systems.png")}<figcaption>Fig. 5. Correct and hallucinated responses by system (all 120 clips per condition). Bars carry their values, so the chart does not depend on colour alone.</figcaption></figure>
<figure>{img("C_decomp.png")}<figcaption>Fig. 6. Ablation by language on clean audio: which transcript the response stage sees (gold, whisper-small, turbo, IndicConformer) × which response stage runs.</figcaption></figure>
<figure>{img("B_wer.png","62%")}<figcaption>Fig. 7. ASR word error rate by language and condition. Turbo roughly halves Hindi WER but barely moves Tamil; the Indic-specialised IndicConformer cuts WER substantially in both languages and under noise.</figcaption></figure>
<p><b>Findings.</b></p><ol>
<li><b>The best system is IndicConformer + NLLB grounding, and it is far better than the baseline.</b> On clean audio, correct responses rise from {G("base","clean","all","correct")*100:.0f}% to {G("conf+mt","clean","all","correct")*100:.0f}% and hallucination falls from {G("base","clean","all","halluc")*100:.0f}% to {G("conf+mt","clean","all","halluc")*100:.0f}%. {("At 5 dB noise, correct rises from %.0f%% to %.0f%%." % (G("base","noisy","all","correct")*100, G("conf+mt","noisy","all","correct")*100)) if _has("conf+mt","noisy") else ("The LLM stage on noisy audio did not finish in time; the no-LLM control (IndicConformer → NLLB) reaches %.0f%% correct at 5 dB against %.0f%% for the baseline." % (G("mtonly-conf","noisy","all","correct")*100, G("base","noisy","all","correct")*100))} The bootstrap intervals in 2.2 do not overlap between the baseline and the best system.</li>
<li><b>The ASR model is the biggest single lever, and a specialised model beats general ones.</b> Clean-audio WER falls from {G("base","clean","all","WER"):.0f}% (whisper-small) to {G("turbo","clean","all","WER"):.0f}% (whisper-large-v3-turbo) to {G("conf","clean","all","WER"):.0f}% (IndicConformer); Hindi {G("base","clean","hi","WER"):.0f} → {G("turbo","clean","hi","WER"):.0f} → {G("conf","clean","hi","WER"):.0f}%, Tamil {G("base","clean","ta","WER"):.0f} → {G("turbo","clean","ta","WER"):.0f} → {G("conf","clean","ta","WER"):.0f}%. Turbo barely helps Tamil; IndicConformer does. It is also by far the most noise-robust: at 5 dB white noise its WER is {G("mtonly-conf","noisy","all","WER"):.0f}% against {G("turbo","noisy","all","WER"):.0f}% for turbo and {G("base","noisy","all","WER"):.0f}% for whisper-small.</li>
<li><b>Grounding and ASR are complementary, and grounding is what rescues Tamil.</b> With IndicConformer text, plain Qwen reaches only {G("conf","clean","ta","correct")*100:.0f}% correct on Tamil but {G("conf","clean","hi","correct")*100:.0f}% on Hindi; adding the NLLB draft lifts them to {G("conf+mt","clean","ta","correct")*100:.0f}% and {G("conf+mt","clean","hi","correct")*100:.0f}%. With the weaker whisper-small transcripts the same grounding takes Tamil from 2% to 37%.</li>
<li><b>Negative finding: on this task the LLM adds essentially nothing over the MT model.</b> Returning NLLB&rsquo;s translation directly (no LLM) scores {G("mtonly-conf","clean","all","correct")*100:.0f}% correct with IndicConformer against {G("conf+mt","clean","all","correct")*100:.0f}% for NLLB + Qwen, a difference well inside the noise at n = 120, and the same pattern held for whisper-small and turbo. The gain from &ldquo;grounding&rdquo; is therefore the dedicated MT model replacing the LLM&rsquo;s weak Indic reading, and I do not present it as an LLM improvement. Whether the LLM helps when the task needs reasoning is tested in 2.7, where it is essential.</li>
<li><b>This metric is near its ceiling.</b> Feeding <i>gold</i> transcripts through NLLB gives {G("mtonly-oracle","oracle","all","correct")*100:.0f}% correct, so the best system ({G("conf+mt","clean","all","correct")*100:.0f}%) is within a few points of the best this proxy can show, and differences among the top systems are not resolvable here. That is why I added an exact-answer task (2.7).</li>
<li><b>Tamil is still the harder language.</b> IndicConformer&rsquo;s Tamil WER ({G("conf","clean","ta","WER"):.0f}%) is more than twice its Hindi WER ({G("conf","clean","hi","WER"):.0f}%), partly because Tamil is agglutinative (CER {G("conf","clean","ta","CER"):.0f}% vs WER); on the comprehension task (2.7) Tamil stays close to chance even with gold text.</li></ol>

<h3>2.5 Latency</h3>
<p>Measured in a separate benchmark on a fixed 24-clip clean subset (12 Hindi + 12 Tamil, {A:.0f} s of audio), one model in memory at a time. <b>The laptop was not idle:</b> swap usage at the start was <small>{html.escape(lat['swap_at_start'])}</small>, i.e. memory pressure from other applications. Absolute times are therefore probably pessimistic, and a quiet machine should be faster; the comparison <i>between</i> pipelines is the reliable part, because every stage ran under the same conditions. Stage times are additive, so each pipeline&rsquo;s latency is the sum of its stages.</p>
<table><tr><th>Pipeline</th><th>Mean latency per utterance (s)</th><th>RTF</th></tr>{lat_rows}</table>
<table><tr><th>Stage</th><th>Mean seconds per utterance</th></tr>{stage_rows}</table>
<p>All pipelines run faster than real time (RTF &lt; 1) on this laptop, and turbo is <i>faster</i> than whisper-small despite being larger because it decodes with a 4-layer decoder. Grounding costs about one extra NLLB call. IndicConformer runs on CPU through ONNX (no MPS), so it is not in this GPU benchmark; in the full evaluation run its ASR RTF was {G('conf','clean','all','asr_RTF'):.2f} (clean), i.e. about 18× faster than real time, but that figure was taken under memory pressure and with the CPU shared, so treat it as indicative. These are batch (non-streaming) offline numbers: the whole utterance must be heard before any processing starts, so end-to-end <i>response latency after the user stops speaking</i> is the figure above, not a streaming first-token latency. The timing columns in <code>results/summary.csv</code> come from the full evaluation runs, during which memory pressure varied a lot (one run was visibly distorted by swapping), so use this benchmark rather than those columns.</p>

<h3>2.6 Threats to validity</h3>
<ul><li><b>Small sample, one seed.</b> 120 utterances (60 per language); intervals are wide for per-language cells (n = 60).</li>
<li><b>Proxy metrics</b> (see 2.1). No human audit.</li>
<li><b>Read speech.</b> FLEURS is read, clean-studio-ish speech; spontaneous Indic speech will be harder. Noise is synthetic white noise, harsher and less structured than real babble or street noise.</li>
<li><b>Task choice.</b> Translation is a narrow notion of &ldquo;response&rdquo;; it was chosen so correctness is measurable. The LLM-adds-little finding is specific to this task.</li>
<li><b>Thresholds were fixed after viewing outputs</b> (disclosed above); I re-scored correctness at 0.5 and 0.7 (<code>correct_0.5</code>, <code>correct_0.7</code> in <code>results/per_utterance.csv</code>). Across the nine systems, the coarse ordering is stable at 0.5, 0.6 and 0.7: on clean audio the four best (IndicConformer + NLLB + Qwen, its no-LLM control, turbo + NLLB + Qwen, its control) and the two worst (turbo + Qwen, baseline) keep their places, while the middle ranks (plain IndicConformer + Qwen, and the whisper-small + NLLB variants) swap. On noisy audio the three IndicConformer systems are always the top three, followed by the turbo systems; the three whisper-small systems all score ≤ 11% and trade places, and IndicConformer + NLLB + Qwen and its no-LLM control swap first and second, so those two should be read as indistinguishable.</li>
<li><b>Language is forced</b> for Whisper (known-language setting); automatic language ID was not evaluated.</li></ul>

<h3 style="page-break-before:always">Appendix: example outputs</h3>
<p><small>The first two utterances of each language in manifest order (no selection by outcome).</small></p>{ex}
"""

qa = pd.read_csv("results/qa_summary.csv"); qc = pd.read_csv("results/qa_contrasts.csv"); qb = pd.read_csv("results/qa_buckets.csv")
QN = {"gold": "Gold transcript (upper bound)", "small": "whisper-small", "turbo": "whisper-large-v3-turbo", "conf": "IndicConformer (RNNT)"}
QM = {"native": "native-language prompt", "en": "NLLB-translated to English"}
def qv(src, mode, lg, col): return float(qa[(qa.source == src) & (qa["mode"] == mode) & (qa.lang == lg)][col].iloc[0])
def qcell(src, mode, lg): return f"{qv(src,mode,lg,'acc'):.0f} <span class=ci>[{qv(src,mode,lg,'lo'):.0f}–{qv(src,mode,lg,'hi'):.0f}]</span>"
def qcon(label, lg="all"):
    r = qc[(qc.contrast == label) & (qc.lang == lg)].iloc[0]; return f"{r.diff_pts:+.1f} pts [{r.lo:+.1f}, {r.hi:+.1f}]"
qa_rows = ""
for src in ("gold", "small", "turbo", "conf"):
    for mode in ("native", "en"):
        w = "" if src == "gold" else f"{qv(src,mode,'hi','passage_WER'):.0f} / {qv(src,mode,'ta','passage_WER'):.0f}"
        qa_rows += (f"<tr><td>{QN[src]}</td><td>{QM[mode]}</td><td>{qcell(src,mode,'hi')}</td><td>{qcell(src,mode,'ta')}</td>"
                    f"<td><b>{qcell(src,mode,'all')}</b></td><td>{w}</td></tr>")
qa_table = ("<table><tr><th>Passage source</th><th>Prompt</th><th>Hindi acc. % [95% CI]</th><th>Tamil acc. % [95% CI]</th><th>All (n=200)</th>"
            "<th>Passage WER % (hi / ta)</th></tr>" + qa_rows + "</table>")
cw = [qv(sr, md, "all", "confident_wrong") for sr in ("gold", "small", "turbo", "conf") for md in ("native", "en")]
T3 = f"""
<h3>2.7 Listening comprehension: an exact-answer response task (Belebele-Fleurs)</h3>
<p>The translation task in 2.1&ndash;2.6 relies on proxy metrics, and the LLM turned out to add little there. To get a response task with an <b>exact</b> answer and one where the LLM has to reason, I added a spoken-passage multiple-choice test, built on <i>Belebele-Fleurs</i> (WueNLP/belebele-fleurs, CC-BY-SA): a FLORES passage is <b>spoken</b> (one FLEURS recording per sentence), and a <b>written</b> four-option question about it is asked. Pipeline: passage audio &rarr; ASR &rarr; (optionally NLLB translation to English) &rarr; Qwen2.5-3B, which picks the option letter. The letter is chosen as the argmax of the model's next-token probabilities over A/B/C/D (no sampling), so scoring is deterministic and <b>accuracy is exact</b>; chance is 25%.</p>
<ul><li><b>Sample.</b> The first 100 questions per language in the dataset's own order (Hindi: first shard; Tamil: first two shards), i.e. 200 questions over 424 distinct sentence clips (92.7 min of audio). They are not randomly sampled, and the sample is small (95% Wilson intervals are about &plusmn;9 points per language).</li>
<li><b>Conditions.</b> Four passage sources: gold text (the ceiling), whisper-small, whisper-large-v3-turbo and IndicConformer-600M (RNNT, chosen in advance). Two prompt modes: passage, question and options in the original language (&ldquo;native&rdquo;), or all translated to English by NLLB-200-600M.</li>
<li><b>Confidently wrong.</b> I report the share of questions where the model picked a wrong option with probability &ge; 0.7, the multiple-choice analogue of a confident hallucination.</li></ul>
{qa_table}
<figure>{img("E_qa.png")}<figcaption>Fig. 8. Answer accuracy by passage source and prompt mode. Error bars are 95% Wilson intervals; the dashed line is chance (25%). With 100 questions per language the intervals are wide.</figcaption></figure>
<p><b>Findings.</b></p><ol>
<li><b>Better ASR improves comprehension, significantly.</b> With English prompts, IndicConformer beats whisper-small by {qcon("Conformer vs whisper-small (English prompt)")} on all 200 questions (paired bootstrap; Hindi {qcon("Conformer vs whisper-small (English prompt)","hi")}, Tamil {qcon("Conformer vs whisper-small (English prompt)","ta")}), and turbo beats whisper-small by {qcon("Turbo vs whisper-small (English prompt)")}. The Tamil interval includes zero, so the effect is established for Hindi and pooled data, not for Tamil alone.</li>
<li><b>The ceiling is the LLM, not the ASR.</b> Even with <i>gold</i> text Qwen2.5-3B answers only {qv("gold","en","all","acc"):.0f}% correctly with English prompts, and for Tamil {qv("gold","native","ta","acc"):.0f}&ndash;{qv("gold","en","ta","acc"):.0f}% against 25% chance. IndicConformer is within {qcon("Gold vs Conformer (English prompt)")} of that ceiling (interval includes zero), so there is little left for ASR to gain on this model. This matches the translation experiment, where Tamil was also capped by the LLM.</li>
<li><b>Lower WER does not always buy more accuracy.</b> IndicConformer has far lower passage WER than turbo (Hindi {qv("conf","en","hi","passage_WER"):.0f}% vs {qv("turbo","en","hi","passage_WER"):.0f}%, Tamil {qv("conf","en","ta","passage_WER"):.0f}% vs {qv("turbo","en","ta","passage_WER"):.0f}%), but the two reach similar accuracy ({qv("conf","en","all","acc"):.0f}% vs {qv("turbo","en","all","acc"):.0f}%, English prompts): once WER is below roughly 30%, the LLM&rsquo;s comprehension is the binding constraint.</li>
<li><b>Translating to English helps a little.</b> English prompts score higher than native prompts in every pairing ({qcon("English prompt vs native prompt (gold text)")} with gold text, {qcon("English prompt vs native prompt (Conformer)")} with IndicConformer), but these intervals include zero at n = 200, so I treat it as suggestive rather than established.</li>
<li><b>Models are confidently wrong.</b> Depending on source and mode, {min(cw):.0f}&ndash;{max(cw):.0f}% of all questions are answered wrongly with probability &ge; 0.7; this stays high even with gold text, so it reflects model overconfidence rather than ASR damage alone.</li></ol>
<p><small>Checks: predicted letters are spread over all four options (D is chosen somewhat less often, about 14&ndash;18% of the time against 20% of gold answers), so accuracy is not an artefact of always answering one letter. Only the transcription stage, not the question, is spoken; a fully spoken question-and-answer test, other LLMs, and larger samples are left for future work.</small></p>
"""

head = ("<!doctype html><html><head><meta charset=utf-8><title>Speech LLM architectures and an Indic speech pipeline</title><style>" + CSS + "</style></head><body>")
title = ("<h1>Speech LLM architectures, and an evaluated Indic speech → text-response pipeline</h1>"
         "<p><small>Technical assignment, Speech AI Research Internship, Vaani Research Lab. Code and reproduction steps: README in the repository.</small></p>")
T2 = T2.replace("<h3>2.6 Threats to validity</h3>", T3 + "<h3>2.8 Threats to validity</h3>")
T2 = T2.replace("<h3>2.3 Bottleneck analysis</h3>", "<h3>ASR systems compared</h3><p>WER / CER (%), corpus level, on the 120 translation-set clips (60 per language). IndicConformer was run with both of its decoders; RNNT was fixed in advance for the downstream stages.</p>" + _asr_rows() + "<h3>2.3 Bottleneck analysis</h3>")
Path("report/report.html").write_text(head + title + t1 + T2 + "</body></html>", encoding="utf-8")
print("wrote report/report.html")
