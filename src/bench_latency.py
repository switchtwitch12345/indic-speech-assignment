"""Stage-level latency on a fixed 24-clip subset (12 hi + 12 ta, clean), one model resident at a time.
Stage times are additive, so any pipeline's latency = sum of its stages. Records swap usage at start for transparency.
usage: python src/bench_latency.py"""
import gc, json, subprocess, time
import numpy as np, torch
from data import load
from audio import load_wav
from asr import Whisper, sync

items = [x for lg in ("hi", "ta") for x in [i for i in load() if i["lang"] == lg][:12]]
dur = sum(i["duration"] for i in items)
swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
res = dict(n=len(items), audio_s=dur, swap_at_start=swap, stages={})
free = lambda: (gc.collect(), torch.mps.empty_cache())
hyps = {}
for tag, name, dt in [("whisper-small", "openai/whisper-small", torch.float32),
                      ("whisper-large-v3-turbo", "openai/whisper-large-v3-turbo", torch.float16)]:
    m = Whisper(name, dt); m.transcribe(load_wav(items[0]["wav"])[:48000], "hi")
    ts = []; hyps[tag] = []
    for it in items:
        h, t = m.transcribe(load_wav(it["wav"]), it["lang"]); ts.append(t); hyps[tag].append(h)
    res["stages"][tag] = dict(total_s=sum(ts), rtf=sum(ts) / dur, mean_s=float(np.mean(ts)))
    del m; free()
from nllb import NLLB
mt = NLLB(); mt.translate("नमस्ते", "hi"); drafts = {}; ts = []
for tag in hyps:
    drafts[tag] = []
    for it, h in zip(items, hyps[tag]):
        sync(); t0 = time.perf_counter(); drafts[tag].append(mt.translate(h, it["lang"])); sync(); ts.append(time.perf_counter() - t0)
res["stages"]["nllb"] = dict(total_s=sum(ts), rtf=sum(ts) / (2 * dur), mean_s=float(np.mean(ts)))
del mt; free()
from llm import LLM
llm = LLM(); llm.respond("नमस्ते", "hi", 8)
for key, use_draft in [("qwen_plain", False), ("qwen_grounded", True)]:
    ts = []
    for it, h, d in zip(items, hyps["whisper-large-v3-turbo"], drafts["whisper-large-v3-turbo"]):
        _, t, _ = llm.respond(h, it["lang"], draft=d if use_draft else None); ts.append(t)
    res["stages"][key] = dict(total_s=sum(ts), rtf=sum(ts) / dur, mean_s=float(np.mean(ts)))
json.dump(res, open("results/latency_bench.json", "w"), indent=1)
print(json.dumps(res, indent=1))
