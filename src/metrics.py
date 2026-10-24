"""Self-contained evaluation: text normalisation, WER/CER, response correctness, hallucination."""
import re, unicodedata, random
import numpy as np, jiwer, sacrebleu

_ZW = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def norm(s):
    """NFC, drop zero-width chars, lowercase, strip all punctuation/symbols (incl. danda), squeeze spaces."""
    s = unicodedata.normalize("NFC", s).translate(_ZW).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


def wer(refs, hyps):
    return jiwer.wer([norm(r) for r in refs], [norm(h) or "<empty>" for h in hyps])


def cer(refs, hyps):
    return jiwer.cer([norm(r) for r in refs], [norm(h) or "<empty>" for h in hyps])


def utt_wer(r, h):
    return jiwer.wer(norm(r), norm(h) or "<empty>")


def bootstrap_ci(refs, hyps, fn, n=1000, seed=0):
    rng = random.Random(seed); idx = list(range(len(refs))); vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        vals.append(fn([refs[i] for i in s], [hyps[i] for i in s]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---- response quality -------------------------------------------------------
_emb = None


def embed(texts):
    global _emb
    if _emb is None:
        from sentence_transformers import SentenceTransformer
        _emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _emb.encode(texts, normalize_embeddings=True, batch_size=64)


def chrf(hyps, refs):
    return sacrebleu.corpus_chrf(hyps, [refs]).score


_NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
         "eight": "8", "nine": "9", "ten": "10"}


def numbers(s):
    toks = re.findall(r"[a-z]+|\d[\d,\.]*", s.lower())
    return {_NUMW.get(t, t.replace(",", "")) for t in toks if t in _NUMW or t[0].isdigit()}


def hallucination_flags(resp, ref_en, cos, tau_h=0.4, max_ratio=2.0):
    """Three transparent, reference-based hallucination criteria (see README).
    H1 numeric: response states a number absent from the reference translation.
    H2 drift  : semantic similarity to the reference below tau_h (content unrelated to what was said).
    H3 verbose: response is >max_ratio x longer than the reference (+5 words slack) -> content added beyond a translation.
    """
    h1 = bool(numbers(resp) - numbers(ref_en))
    h2 = cos < tau_h
    h3 = len(resp.split()) > max_ratio * len(ref_en.split()) + 5
    return h1, h2, h3
