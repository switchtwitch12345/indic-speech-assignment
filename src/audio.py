import numpy as np, librosa

SR = 16000


def load_wav(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def add_noise(y, snr_db, seed):
    """Add white Gaussian noise at a fixed SNR (dB), deterministic per utterance."""
    rng = np.random.default_rng(seed)
    p_sig = np.mean(y ** 2) + 1e-12
    noise = rng.standard_normal(len(y)).astype(np.float32)
    noise *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / np.mean(noise ** 2))
    return np.clip(y + noise, -1, 1)
