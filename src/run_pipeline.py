"""Speech -> ASR -> LLM -> text response. Caches per (system, condition) to results/*.jsonl.

usage: python src/run_pipeline.py --asr openai/whisper-small --tag base --conds clean noisy
"""
import argparse, json, hashlib, time
from pathlib import Path
from tqdm import tqdm
from data import load
from audio import load_wav, add_noise
from asr import Whisper, sync
from llm import LLM
from nllb import NLLB

SNR = 5.0


def seed_of(uid):
    return int(hashlib.md5(uid.encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="openai/whisper-small")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--conds", nargs="+", default=["clean", "noisy"])
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--ground", action="store_true", help="NLLB translation draft is given to the LLM")
    ap.add_argument("--oracle", action="store_true", help="also run LLM on gold transcripts")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    a = ap.parse_args()
    items = load(a.manifest)
    asr, llm = Whisper(a.asr), LLM(a.llm)
    mt = NLLB() if a.ground else None
    # warm-up so first-item latency isn't dominated by kernel compilation
    asr.transcribe(load_wav(items[0]["wav"])[:16000 * 3], "hi"); llm.respond("नमस्ते", "hi", 8)

    if a.oracle:
        out = Path("results/oracle_ground.jsonl" if a.ground else "results/oracle.jsonl")
        if not out.exists():
            with open(out, "w", encoding="utf-8") as f:
                for it in tqdm(items, desc="oracle"):
                    r, dt, n = llm.respond(it["ref_text"], it["lang"], draft=mt.translate(it["ref_text"], it["lang"]) if mt else None)
                    f.write(json.dumps(dict(uid=it["uid"], cond="oracle", hyp_text=it["ref_text"], response=r,
                                            t_llm=dt, n_tok=n), ensure_ascii=False) + "\n")
    for cond in a.conds:
        out = Path(f"results/{a.tag}_{cond}.jsonl")
        if out.exists():
            print("skip", out); continue
        with open(out, "w", encoding="utf-8") as f:
            for it in tqdm(items, desc=f"{a.tag}/{cond}"):
                y = load_wav(it["wav"])
                if cond == "noisy":
                    y = add_noise(y, SNR, seed_of(it["uid"]))
                hyp, t_asr = asr.transcribe(y, it["lang"], a.beams)
                if mt:
                    sync(); t1 = time.perf_counter(); draft = mt.translate(hyp, it["lang"]); sync()
                    t_mt = time.perf_counter() - t1
                else:
                    draft, t_mt = None, 0.0
                resp, t_llm, n = llm.respond(hyp, it["lang"], draft=draft)
                t_llm += t_mt  # MT cost is charged to the language-model stage
                f.write(json.dumps(dict(uid=it["uid"], cond=cond, hyp_text=hyp, response=resp, t_asr=t_asr,
                                        t_llm=t_llm, n_tok=n, dur=it["duration"]), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
