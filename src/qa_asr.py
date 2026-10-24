"""Stage 1 of the listening-comprehension eval: transcribe each unique passage sentence once.
usage: python src/qa_asr.py --tag small --asr openai/whisper-small
       python src/qa_asr.py --tag conf --asr ai4bharat/indic-conformer-600m-multilingual --decoder rnnt
       python src/qa_asr.py --tag gold            (gold transcripts, no ASR)"""
import argparse, json
from pathlib import Path
import torch
from tqdm import tqdm
import belebele
from audio import load_wav

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--asr"); ap.add_argument("--decoder", default="rnnt")
ap.add_argument("--dtype", default="fp32")
a = ap.parse_args()
sents = {}
for q in belebele.load():
    for s in q["sentences"]: sents[s["sid"]] = s
out = Path(f"results/qa/asr_{a.tag}.jsonl")
if out.exists(): print("skip", out); raise SystemExit(0)
if a.tag == "gold":
    rows = [dict(sid=k, hyp=s["ref"], t=0.0, dur=s["dur"]) for k, s in sents.items()]
else:
    from asr import Whisper, IndicConformer
    m = IndicConformer(decoder=a.decoder) if "conformer" in a.asr else \
        Whisper(a.asr, torch.float16 if a.dtype == "fp16" else torch.float32)
    k0 = next(iter(sents)); m.transcribe(load_wav(sents[k0]["wav"])[:48000], k0.split("_")[0])  # warm-up
    rows = []
    for k, s in tqdm(sents.items(), desc=f"asr/{a.tag}"):
        h, t = m.transcribe(load_wav(s["wav"]), k.split("_")[0])
        rows.append(dict(sid=k, hyp=h, t=t, dur=s["dur"]))
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
print(out, len(rows))
