"""Belebele-Fleurs listening-comprehension manifest (WueNLP/belebele-fleurs).
A spoken FLORES passage (one FLEURS recording per sentence) + a written multiple-choice question about it.
Answers are exact, so accuracy replaces the proxy correctness metric used for the translation task.
usage: python src/belebele.py [n_per_lang]"""
import io, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa, pyarrow.parquet as pq

RAW = Path("data/belebele/raw/data"); WAV = Path("data/belebele/wav"); OUT = "data/belebele/manifest.jsonl"
SHARDS = {"hi": ["hin_Deva/test-00000-of-00005.parquet"],
          "ta": ["tam_Taml/test-00000-of-00008.parquet", "tam_Taml/test-00001-of-00008.parquet"]}
COLS = ["question_number", "flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4",
        "correct_answer_num", "sentence_data", "link"]


def save_wav(b, path):
    y, sr = sf.read(io.BytesIO(b), dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sf.write(path, y, 16000)
    return len(y) / 16000


def build(n_per_lang=100):
    WAV.mkdir(parents=True, exist_ok=True); rows = []
    for lg, shards in SHARDS.items():
        k = 0
        for sh in shards:
            f = pq.ParquetFile(RAW / sh)
            for g in range(f.num_row_groups):
                for r in f.read_row_group(g, columns=COLS).to_pylist():
                    if k >= n_per_lang: break
                    sents = []
                    for s in r["sentence_data"]:
                        if not s["audio"]: break
                        wp = WAV / f"{lg}_{s['fleurs_id']}.wav"
                        dur = save_wav(s["audio"][0]["bytes"], wp) if not wp.exists() else sf.info(wp).duration
                        sents.append(dict(sid=f"{lg}_{s['fleurs_id']}", wav=str(wp), ref=s["raw_transcription"], dur=dur))
                    else:
                        rows.append(dict(qid=f"{lg}_{k}", lang=lg, passage=r["flores_passage"], sentences=sents,
                                         question=r["question"], options=[r[f"mc_answer{i}"] for i in range(1, 5)],
                                         answer=int(r["correct_answer_num"]) - 1, link=r["link"]))
                        k += 1
        print(lg, k, "questions")
    Path(OUT).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    uniq = {s["sid"]: s["dur"] for r in rows for s in r["sentences"]}
    print(f"{len(rows)} questions, {len(uniq)} unique sentence clips, {sum(uniq.values())/60:.1f} min of audio ->", OUT)


def load(path=OUT):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
