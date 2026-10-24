import time, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LANG = {"hi": "hindi", "ta": "tamil"}


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


class Whisper:
    def __init__(self, name="openai/whisper-small", dtype=torch.float32):
        self.name = name
        self.proc = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(DEV).eval()
        self.dtype = dtype

    @torch.inference_mode()
    def transcribe(self, y, lang, num_beams=1):
        """Returns (text, seconds). Language is forced (known-language setting, same for all systems)."""
        sync(); t0 = time.perf_counter()
        feats = self.proc(y, sampling_rate=16000, return_tensors="pt").input_features.to(DEV, self.dtype)
        ids = self.model.generate(feats, language=LANG[lang], task="transcribe", num_beams=num_beams,
                                  max_new_tokens=224)
        text = self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        sync()
        return text, time.perf_counter() - t0


class IndicConformer:
    """AI4Bharat IndicConformer-600M multilingual (hybrid CTC/RNNT; ONNX + TorchScript, runs on CPU).
    Loaded from a real (non-symlinked) local copy, models/indic-conformer, because recent onnxruntime rejects
    external-data paths that resolve outside the model directory (which is what Hugging Face cache symlinks do).
    Populate it with:  hf download ai4bharat/indic-conformer-600m-multilingual --local-dir models/indic-conformer
    (gated: accept the terms on the model page and `hf auth login` first). Uses the repo's own model_onnx.py."""
    def __init__(self, path="models/indic-conformer", decoder="rnnt"):
        import sys, os
        path = os.path.abspath(path)
        sys.path.insert(0, path)
        from model_onnx import IndicASRModel, IndicASRConfig
        self.name, self.decoder = "ai4bharat/indic-conformer-600m-multilingual", decoder
        self.model = IndicASRModel(IndicASRConfig(ts_folder=path))

    def transcribe(self, y, lang, num_beams=1):
        wav = torch.from_numpy(y).unsqueeze(0)
        t0 = time.perf_counter()
        out = self.model(wav, lang, self.decoder)
        dt = time.perf_counter() - t0
        if isinstance(out, (list, tuple)): out = out[0]
        return str(out).strip(), dt
