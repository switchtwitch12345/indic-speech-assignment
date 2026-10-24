# Indic speech → text-response pipeline, with evaluation

Speech (Hindi, Tamil) → ASR → LLM → English text response, evaluated on 120 FLEURS test utterances under clean and noisy audio.
Everything runs locally on open-source models (developed on an Apple M2 Pro, 16 GB, PyTorch/MPS). The report PDF has the full analysis,
including Task 1 (Moshi vs Qwen3-Omni architecture comparison).

## What it does

| Stage | Baseline | Variants tested |
|---|---|---|
| ASR | `openai/whisper-small` | `openai/whisper-large-v3-turbo`; `ai4bharat/indic-conformer-600m-multilingual` (CTC and RNNT) |
| LLM | `Qwen/Qwen2.5-3B-Instruct` | same, optionally given an NLLB translation draft |
| MT (optional) | – | `facebook/nllb-200-distilled-600M` (grounding, and as a no-LLM control) |

Two response tasks, both Hindi + Tamil:
1. **Say in English what the speaker said** (120 FLEURS clips, clean and 5 dB noise). FLEURS audio is aligned to FLORES English, so a reference exists for every clip. Correctness and hallucination are automatic proxies.
2. **Spoken-passage multiple choice** (Belebele-Fleurs, 100 questions per language): the passage is spoken, the question is written; the answer is exact, so accuracy needs no proxy.

## Headline results

Translation task, 120 clips (see the report for per-language results, 95% CIs and ablations):

| System | Audio | WER % | CER % | Correct % | Hallucinated % |
|---|---|---|---|---|---|
| whisper-small → Qwen (baseline) | clean | 69.3 | 27.6 | 9 | 74 |
| IndicConformer → NLLB → Qwen (best) | clean | 19.4 | 8.8 | 91 | 8 |
| whisper-small → Qwen (baseline) | 5 dB noise | 97.7 | 56.9 | 4 | 94 |
| IndicConformer → NLLB → Qwen (best) | 5 dB noise | 28.1 | 14.4 | 75 | 17 |

Listening comprehension, 200 questions (chance 25%): gold text 56%, IndicConformer 52%, turbo 53%, whisper-small 40% (English-translated prompts).

Findings: the ASR model is the biggest lever and an Indic-specialised model beats larger general ones, especially under noise. Hindi is limited by ASR with the baseline; Tamil is limited by the LLM's comprehension (with gold Tamil text Qwen2.5-3B is near chance on the multiple-choice task). On the translation task a control that returns the NLLB translation *without* the LLM scores about the same as NLLB + LLM, so the gain there comes from the MT model, not LLM reasoning. Correctness/hallucination on the translation task are automatic proxies with thresholds I set by inspection (not human-calibrated).

## Metric definitions (all implemented in `src/metrics.py`)

- **WER / CER**: reference = FLEURS raw transcript; both sides NFC-normalised, lower-cased, punctuation (incl. danda) stripped. Corpus-level; bootstrap 95% CI.
- **Correct**: cosine(all-MiniLM-L6-v2) between response and English reference ≥ 0.6. Also reported: chrF, and sensitivity at 0.5 / 0.7.
- **Hallucinated**: any of (H1) a number not in the reference, (H2) cosine < 0.4, (H3) response > 2× reference length + 5 words.
- **Latency / RTF**: per-stage wall clock with MPS sync, batch size 1; RTF = processing time / audio duration.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio transformers datasets soundfile librosa jiwer accelerate sentencepiece \
            matplotlib pandas numpy scipy sacrebleu tqdm huggingface_hub sentence-transformers \
            onnxruntime onnx pyarrow pymupdf pypdf

# 1. translation-task data: FLEURS test tsv + audio for hi_in, ta_in, en_us (en_us only supplies the aligned English text)
python - <<'PY'
from huggingface_hub import hf_hub_download
for l in ["hi_in", "ta_in", "en_us"]:
    for f in [f"data/{l}/test.tsv", f"data/{l}/audio/test.tar.gz"]:
        hf_hub_download("google/fleurs", f, repo_type="dataset", local_dir="data/raw")
PY
for l in hi_in ta_in en_us; do mkdir -p data/raw/data/$l/wav && tar xzf data/raw/data/$l/audio/test.tar.gz -C data/raw/data/$l/wav; done
PYTHONPATH=src python src/data.py 60                                   # manifest (60 clips per language, seed 0)

# 2. baseline + gold-transcript oracle (clean and 5 dB noise)
PYTHONPATH=src python src/run_pipeline.py --tag base --conds clean noisy --oracle

# 3. gold-transcript source for the oracle variants; whisper-turbo + NLLB variants and controls
PYTHONPATH=src python - <<'PY'
import json; from data import load
with open("results/gold_oracle.jsonl", "w", encoding="utf-8") as f:
    for it in load():
        f.write(json.dumps(dict(uid=it["uid"], cond="oracle", hyp_text=it["ref_text"], t_asr=None, dur=it["duration"]), ensure_ascii=False) + "\n")
PY
./run_all_variants.sh

# 4. IndicConformer (GATED: accept the terms on https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual, then `hf auth login`)
hf download ai4bharat/indic-conformer-600m-multilingual --local-dir models/indic-conformer   # real files, not cache symlinks (onnxruntime rejects symlinks)

# 5. Belebele-Fleurs: download one Hindi and two Tamil shards, build the 200-question manifest
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["data/hin_Deva/test-00000-of-00005.parquet", "data/tam_Taml/test-00000-of-00008.parquet", "data/tam_Taml/test-00001-of-00008.parquet"]:
    hf_hub_download("WueNLP/belebele-fleurs", f, repo_type="dataset", local_dir="data/belebele/raw")
PY
PYTHONPATH=src python src/belebele.py 100

# 6. IndicConformer translation-set runs + the whole listening-comprehension eval (stages skip finished outputs)
./run_qa_and_conformer.sh        # (I ran its tail via run_rest.sh, which orders the same stages by priority)

# 7. latency benchmark, metrics, figures, report
PYTHONPATH=src python src/bench_latency.py
PYTHONPATH=src python src/analyze.py base base+mt turbo turbo+mt mtonly-base mtonly-turbo mtonly-oracle conf conf+mt mtonly-conf
PYTHONPATH=src python src/make_figs.py
PYTHONPATH=src python src/qa_report.py
./report/make_pdf.sh             # needs Google Chrome
```

Notes: every stage keeps one model in memory (16 GB machine). First run downloads about 13 GB of weights (+ ~0.8 GB Belebele-Fleurs shards).
Greedy decoding / argmax scoring and fixed seeds make results repeatable on the same hardware; MPS kernels may differ slightly across machines.
Two early lessons are worth knowing: (i) the multiple-choice scorer asks the LLM for logits of the last position only (`logits_to_keep=1`); computing the full prompt's logits made long Tamil prompts blow memory and slowed one stage ~40×; (ii) `progress.py` shows live progress.

## Layout

```
src/data.py            manifest builder (FLEURS + FLORES-aligned English)
src/asr.py llm.py nllb.py     model wrappers (Whisper, IndicConformer, Qwen, NLLB)    src/audio.py   loading, seeded noise
src/run_pipeline.py    baseline pipeline + oracle     src/run_asr.py / respond.py / mt_only.py   staged variants
src/belebele.py        listening-comprehension manifest   src/qa_asr.py qa_mt.py qa_answer.py qa_report.py   its stages + analysis
src/metrics.py         WER/CER, correctness, hallucination, bootstrap
src/analyze.py         aggregate -> results/summary.csv, per_utterance.csv
src/bench_latency.py   isolated stage-level latency   src/make_figs.py   charts
results/               per-utterance outputs (jsonl), summaries, latency benchmark
report/                task1.html, figs.py (diagrams), build_report.py, report PDF
```

## Limitations

120 translation clips and 200 multiple-choice questions (100 per language, first-N not random), one seed; read (not spontaneous) speech; synthetic white noise; automatic proxy metrics with thresholds set by inspection;
Whisper's language is forced rather than detected; the translation-style response is a narrow notion of "response" (the multiple-choice task is spoken-passage, written-question). Details in the report.
