"""ASR-only stage (one model resident at a time -> low memory). Writes results/<tag>_<cond>.jsonl with hyp_text + t_asr.
usage: python src/run_asr.py --asr openai/whisper-large-v3-turbo --tag turbo --dtype fp16 [--beams 1]"""
import argparse, json
from pathlib import Path
import torch
from tqdm import tqdm
from data import load
from audio import load_wav, add_noise
from asr import Whisper, IndicConformer
from run_pipeline import seed_of, SNR

ap = argparse.ArgumentParser()
ap.add_argument("--asr", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--conds", nargs="+", default=["clean", "noisy"]); ap.add_argument("--beams", type=int, default=1)
ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
ap.add_argument("--decoder", default="rnnt", choices=["ctc", "rnnt"], help="IndicConformer only")
a = ap.parse_args()
items = load()
asr = IndicConformer(decoder=a.decoder) if "indic-conformer" in a.asr else \
    Whisper(a.asr, torch.float16 if a.dtype == "fp16" else torch.float32)
asr.transcribe(load_wav(items[0]["wav"])[:16000 * 3], "hi")  # warm-up
for cond in a.conds:
    out = Path(f"results/{a.tag}_{cond}.jsonl")
    if out.exists():
        print("skip", out); continue
    with open(out, "w", encoding="utf-8") as f:
        for it in tqdm(items, desc=f"{a.tag}/{cond}"):
            y = load_wav(it["wav"])
            if cond == "noisy":
                y = add_noise(y, SNR, seed_of(it["uid"]))
            hyp, t = asr.transcribe(y, it["lang"], a.beams)
            f.write(json.dumps(dict(uid=it["uid"], cond=cond, hyp_text=hyp, t_asr=t, dur=it["duration"]),
                               ensure_ascii=False) + "\n")
