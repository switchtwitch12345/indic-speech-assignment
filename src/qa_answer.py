"""Stage 3: Qwen answers each multiple-choice question from the (transcribed or gold) passage.
mode 'native': passage + question + options in the original language. mode 'en': NLLB-translated passage, question, options.
usage: python src/qa_answer.py --tags gold small turbo conf"""
import argparse, gc, json
import torch
from pathlib import Path
from tqdm import tqdm
import belebele
from llm import LLM

ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); a = ap.parse_args()
qs = belebele.load(); qopt = {r["qid"]: r for r in map(json.loads, open("results/qa/mt_qopts.jsonl", encoding="utf-8"))}
llm = LLM(); llm.choose("x", "y", ["a", "b", "c", "d"])
for tag in a.tags:
    hyp = {r["sid"]: r["hyp"] for r in map(json.loads, open(f"results/qa/asr_{tag}.jsonl", encoding="utf-8"))}
    en = {r["sid"]: r["en"] for r in map(json.loads, open(f"results/qa/mt_{tag}.jsonl", encoding="utf-8"))}
    for mode in ("native", "en"):
        out = Path(f"results/qa/ans_{tag}_{mode}.jsonl")
        if out.exists(): print("skip", out); continue
        with open(out, "w", encoding="utf-8") as f:
            for q in tqdm(qs, desc=f"{tag}/{mode}"):
                ids = [s["sid"] for s in q["sentences"]]
                if mode == "native": passage, qq, oo = " ".join(hyp[i] for i in ids), q["question"], q["options"]
                else: passage, qq, oo = " ".join(en[i] for i in ids), qopt[q["qid"]]["question"], qopt[q["qid"]]["options"]
                p, dt = llm.choose(passage, qq, oo)
                if len(f.name) and q["qid"].endswith("0"): gc.collect(); torch.mps.empty_cache()  # keep the GPU cache from growing
                pred = max(range(4), key=p.__getitem__)
                f.write(json.dumps(dict(qid=q["qid"], lang=q["lang"], probs=p, pred=pred, answer=q["answer"],
                                        correct=int(pred == q["answer"]), t_llm=dt), ensure_ascii=False) + "\n")
