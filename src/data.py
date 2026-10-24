"""Build the evaluation manifest: FLEURS hi/ta test clips joined to FLORES-aligned English references."""
import csv, json, random, sys
from pathlib import Path
import soundfile as sf

RAW = Path("data/raw/data")
COLS = ["id", "file", "raw", "transcription", "phones", "n_samples", "gender"]


def read_tsv(lang):
    rows = {}
    with open(RAW / lang / "test.tsv", encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            d = dict(zip(COLS, r))
            rows.setdefault(d["id"], d)  # one speaker per sentence id is enough
    return rows


def build(n_per_lang=60, seed=0, out="data/manifest.jsonl"):
    en = read_tsv("en_us")
    rng = random.Random(seed)
    items = []
    for lang, code in [("hi_in", "hi"), ("ta_in", "ta")]:
        rows = read_tsv(lang)
        ids = sorted(i for i in rows if i in en)
        rng.shuffle(ids)
        for i in ids[:n_per_lang]:
            r = rows[i]
            wav = RAW / lang / "wav" / "test" / r["file"]
            if not wav.exists():
                continue
            dur = sf.info(wav).duration
            items.append(dict(uid=f"{code}_{i}", lang=code, wav=str(wav), ref_text=r["raw"],
                              ref_norm=r["transcription"], ref_en=en[i]["raw"], duration=dur,
                              gender=r["gender"]))
    Path(out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items), encoding="utf-8")
    print(f"{len(items)} items, {sum(x['duration'] for x in items)/60:.1f} min audio ->", out)


def load(path="data/manifest.jsonl"):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
