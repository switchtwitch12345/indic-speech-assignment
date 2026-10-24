"""Stage 2: NLLB Indic->English translation of passage sentences (one set per ASR source) and of questions/options.
usage: python src/qa_mt.py --tags gold small turbo conf"""
import argparse, json
from pathlib import Path
import belebele
from nllb import NLLB

ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); a = ap.parse_args()
qs = belebele.load(); mt = NLLB(); mt.translate("नमस्ते", "hi")
qo = Path("results/qa/mt_qopts.jsonl")
if not qo.exists():
    rows = []
    for lg in ("hi", "ta"):
        sub = [q for q in qs if q["lang"] == lg]
        flat = [q["question"] for q in sub] + [o for q in sub for o in q["options"]]
        tr = mt.translate_batch(flat, lg); n = len(sub)
        for i, q in enumerate(sub):
            rows.append(dict(qid=q["qid"], question=tr[i], options=tr[n + 4 * i: n + 4 * i + 4]))
    qo.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"); print(qo)
for tag in a.tags:
    out = Path(f"results/qa/mt_{tag}.jsonl")
    if out.exists(): print("skip", out); continue
    src = [json.loads(l) for l in open(f"results/qa/asr_{tag}.jsonl", encoding="utf-8")]
    rows = []
    for lg in ("hi", "ta"):
        sub = [r for r in src if r["sid"].startswith(lg + "_")]
        for r, t in zip(sub, mt.translate_batch([r["hyp"] for r in sub], lg)): rows.append(dict(sid=r["sid"], en=t))
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"); print(out, len(rows))
