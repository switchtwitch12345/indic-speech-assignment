import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LANGNAME = {"hi": "Hindi", "ta": "Tamil"}

PROMPT = (
    "Below is an automatic speech-recognition transcript of something a person said in {lang}. "
    "The transcript may contain recognition errors. Reply in English with a faithful translation of what "
    "the speaker said. Do not add information that is not in the transcript. Output only the translation.\n\n"
    "Transcript: {text}"
)


GROUNDED = (
    "Below is an automatic speech-recognition transcript of something a person said in {lang}, together with a "
    "machine-translation draft of it. Both may contain recognition or translation errors. Reply in English with a "
    "faithful, fluent translation of what the speaker said, using the draft as a guide. Do not add information that is "
    "not in the transcript. Output only the translation.\n\nTranscript: {text}\nDraft translation: {draft}"
)


MCQ = (
    "Read the passage and answer the multiple-choice question about it. The passage was produced by speech recognition "
    "or machine translation and may contain errors. Answer with only the letter of the correct option.\n\n"
    "Passage: {passage}\n\nQuestion: {question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n\nAnswer:"
)


class LLM:
    def __init__(self, name="Qwen/Qwen2.5-3B-Instruct"):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(DEV).eval()

    @torch.inference_mode()
    def respond(self, text, lang, max_new_tokens=160, draft=None):
        p = PROMPT.format(lang=LANGNAME[lang], text=text) if draft is None else \
            GROUNDED.format(lang=LANGNAME[lang], text=text, draft=draft)
        msgs = [{"role": "user", "content": p}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(DEV)
        if DEV == "mps": torch.mps.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        if DEV == "mps": torch.mps.synchronize()
        dt = time.perf_counter() - t0
        gen = out[0, ids["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip().split("\n")[0], dt, len(gen)

    @torch.inference_mode()
    def choose(self, passage, question, options):
        """Probability over options A-D from the next-token distribution (no sampling, deterministic). Returns (probs, seconds)."""
        if not hasattr(self, "_letter_ids"):
            self._letter_ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in "ABCD"]
        msgs = [{"role": "user", "content": MCQ.format(passage=passage, question=question, a=options[0], b=options[1],
                                                       c=options[2], d=options[3])}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(DEV)
        if DEV == "mps": torch.mps.synchronize()
        t0 = time.perf_counter()
        logits = self.model(**ids, logits_to_keep=1).logits[0, -1].float()  # only the last position: avoids a (seq x vocab) tensor
        if DEV == "mps": torch.mps.synchronize()
        dt = time.perf_counter() - t0
        return torch.softmax(logits[self._letter_ids], dim=-1).cpu().tolist(), dt
