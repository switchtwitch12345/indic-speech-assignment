import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CODE = {"hi": "hin_Deva", "ta": "tam_Taml"}


class NLLB:
    """Dedicated Indic->English MT model used to ground the LLM (fixes weak Indic comprehension in small LLMs)."""
    def __init__(self, name="facebook/nllb-200-distilled-600M"):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEV).eval()

    @torch.inference_mode()
    def translate(self, text, lang, max_new_tokens=200):
        self.tok.src_lang = CODE[lang]
        x = self.tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEV)
        out = self.model.generate(**x, forced_bos_token_id=self.tok.convert_tokens_to_ids("eng_Latn"),
                                  max_new_tokens=max_new_tokens, num_beams=1)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]

    @torch.inference_mode()
    def translate_batch(self, texts, lang, bs=16, max_new_tokens=200):
        """Batched translation (length-sorted to limit padding). Returns outputs in input order."""
        self.tok.src_lang = CODE[lang]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i])); out = [None] * len(texts)
        for j in range(0, len(order), bs):
            idx = order[j:j + bs]
            x = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEV)
            g = self.model.generate(**x, forced_bos_token_id=self.tok.convert_tokens_to_ids("eng_Latn"),
                                    max_new_tokens=max_new_tokens, num_beams=1)
            for i, t in zip(idx, self.tok.batch_decode(g, skip_special_tokens=True)): out[i] = t
        return out
