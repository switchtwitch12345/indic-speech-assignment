"""LLM stage on top of cached ASR hypotheses. Optionally ground the LLM with an NLLB translation draft.
Stage 1 (NLLB) is run for all items and freed before stage 2 (LLM) loads, so only one model is resident.
usage: python src/respond.py --src asr_turbo --out turbo [--ground]    (src may be 'base', whose rows already hold responses)"""
import argparse, gc, json, time
from pathlib import Path
import torch
from tqdm import tqdm
from data import load

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ground", action="store_true"); ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--conds", nargs="+", default=["clean", "noisy"])
a = ap.parse_args()
items = {x["uid"]: x for x in load()}
rows = {}
for c in a.conds:
    rows[c] = [json.loads(l) for l in open(f"results/{a.src}_{c}.jsonl", encoding="utf-8")]

drafts, t_mt = {}, {}
if a.ground:
    from nllb import NLLB
    from asr import sync
    mt = NLLB()
    mt.translate("नमस्ते", "hi")
    for c, rs in rows.items():
        for r in tqdm(rs, desc=f"nllb/{c}"):
            sync(); t0 = time.perf_counter()
            drafts[(c, r["uid"])] = mt.translate(r["hyp_text"], items[r["uid"]]["lang"]); sync()
            t_mt[(c, r["uid"])] = time.perf_counter() - t0
    del mt; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

from llm import LLM
llm = LLM(a.llm); llm.respond("नमस्ते", "hi", 8)
for c, rs in rows.items():
    out = Path(f"results/{a.out}_{c}.jsonl")
    if out.exists():
        print("skip", out); continue
    with open(out, "w", encoding="utf-8") as f:
        for r in tqdm(rs, desc=f"{a.out}/{c}"):
            it = items[r["uid"]]
            resp, dt, n = llm.respond(r["hyp_text"], it["lang"], draft=drafts.get((c, r["uid"])))
            f.write(json.dumps(dict(uid=r["uid"], cond=c, hyp_text=r["hyp_text"], response=resp, t_asr=r["t_asr"],
                                    t_llm=dt + t_mt.get((c, r["uid"]), 0.0), n_tok=n, dur=r["dur"]),
                               ensure_ascii=False) + "\n")
