"""Control: response = NLLB translation of the ASR hypothesis (no LLM). Shows how much of the grounded gain is MT alone.
usage: python src/mt_only.py --src base --out mtonly-base [--conds clean noisy]"""
import argparse, json
from pathlib import Path
import time
from tqdm import tqdm
from data import load
from nllb import NLLB
from asr import sync

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--conds", nargs="+", default=["clean", "noisy"])
a = ap.parse_args()
items = {x["uid"]: x for x in load()}
mt = NLLB(); mt.translate("नमस्ते", "hi")
for c in a.conds:
    out = Path(f"results/{a.out}_{c}.jsonl")
    if out.exists():
        print("skip", out); continue
    with open(out, "w", encoding="utf-8") as f:
        for l in tqdm(open(f"results/{a.src}_{c}.jsonl", encoding="utf-8"), total=120, desc=f"{a.out}/{c}"):
            r = json.loads(l)
            sync(); t0 = time.perf_counter(); resp = mt.translate(r["hyp_text"], items[r["uid"]]["lang"]); sync()
            f.write(json.dumps(dict(uid=r["uid"], cond=c, hyp_text=r["hyp_text"], response=resp, t_asr=r["t_asr"],
                                    t_llm=time.perf_counter() - t0, n_tok=0, dur=r["dur"]), ensure_ascii=False) + "\n")
