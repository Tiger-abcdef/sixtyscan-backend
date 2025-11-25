# inference_utils.py

import io
import os
from typing import Tuple

import librosa
import numpy as np
from PIL import Image
from pydub import AudioSegment

import torch
from torch import nn
from torchvision import transforms

from model import ParkinsonModel

# ====== CONSTANTS (must match training) ======
SAMPLE_RATE = 16000
N_MELS = 128
IMG_SIZE = 224

# index 0 -> Non-Parkinson, index 1 -> Parkinson
CLASS_NAMES = ["Non-Parkinson", "Parkinson"]

# ==========================================================
# FFmpeg CONFIG (Render uses Linux, NOT Windows .exe)
# ==========================================================

def _get_ffmpeg_path() -> str | None:
    """
    Decide which ffmpeg binary to use.

    Priority:
      1) FFMPEG_PATH env var (if exists and is a file)
      2) /usr/bin/ffmpeg  (typical on Linux servers like Render)
      3) None -> let pydub try system PATH
    """
    env_path = os.getenv("FFMPEG_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    linux_default = "/usr/bin/ffmpeg"
    if os.path.exists(linux_default):
        return linux_default

    return None


FFMPEG_EXECUTABLE = _get_ffmpeg_path()

if FFMPEG_EXECUTABLE:
    AudioSegment.converter = FFMPEG_EXECUTABLE
    # pydub looks at these attributes too – safe to set them all
    AudioSegment.ffmpeg = FFMPEG_EXECUTABLE
    print("[INFO] Using ffmpeg at:", FFMPEG_EXECUTABLE)
else:
    print(
        "[WARN] FFmpeg executable not found automatically. "
        "Pydub will rely on whatever is in system PATH."
    )

# ==========================================================


def audio_bytes_to_mel_image(audio_bytes: bytes) -> Image.Image:
    """
    Accept ANY audio format (webm, wav, mp3, m4a, etc.).
    Convert to WAV using pydub + ffmpeg.
    Then produce a mel spectrogram as a PIL Image.
    """
    # 1) decode audio bytes with pydub (FFmpeg under the hood)
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        # This is where ffmpeg problems will show up clearly
        raise ValueError(f"Could not decode audio: {e}")

    # 2) export to clean WAV in-memory
    wav_buf = io.BytesIO()
    audio.export(wav_buf, format="wav")
    wav_buf.seek(0)

    # 3) librosa loads WAV, resampled
    y, sr = librosa.load(wav_buf, sr=SAMPLE_RATE)

    if y.size == 0:
        raise ValueError("Empty audio after conversion")

    # 4) mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0, 255] as an image
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-8)
    img_array = (S_norm * 255).astype(np.uint8)  # [n_mels, time]

    img = Image.fromarray(img_array)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = img.convert("RGB")

    return img


def _fix_state_dict_keys(state_dict: dict) -> dict:
    """
    Make saved weights compatible with current ParkinsonModel definition.
    Handles older prefixes like 'base.' or 'model.' and renames them
    to 'backbone.' which this class uses.
    """
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("base."):
            new_k = new_k.replace("base.", "backbone.", 1)
        if new_k.startswith("model."):
            new_k = new_k.replace("model.", "backbone.", 1)
        new_state[new_k] = v
    return new_state


def load_model(
    weights_path: str = "best_model.pth",
    device: str = "cpu",
) -> Tuple[nn.Module, torch.device]:
    """
    Load your trained ParkinsonModel with saved weights.
    Used by main.py:
        MODEL, DEVICE = load_model(...)
    """
    dev = torch.device(device)

    model = ParkinsonModel()
    state_dict = torch.load(weights_path, map_location=dev)

    # Fix any old key prefixes
    state_dict = _fix_state_dict_keys(state_dict)

    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    print(f"[inference_utils] Loaded weights from {weights_path} on {dev}")

    return model, dev


@torch.no_grad()
def predict_from_audio_bytes(
    model: nn.Module,
    device: torch.device,
    audio_bytes: bytes,
) -> Tuple[float, str]:
    """
    Main inference helper used by FastAPI (main.py).

    Returns:
        pd_prob: probability of Parkinson (0.0–1.0)
        label:   'Parkinson' or 'Non-Parkinson' from argmax
    """
    # 1) audio -> mel image
    img = audio_bytes_to_mel_image(audio_bytes)

    # 2) same transforms as training
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [H,W,C] -> [C,H,W], [0,1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    x = transform(img).unsqueeze(0).to(device)  # [1,3,224,224]

    # 3) forward pass
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    # IMPORTANT: index 1 = Parkinson
    pd_prob = float(probs[1].item())

    pred_idx = int(torch.argmax(probs).item())
    label = CLASS_NAMES[pred_idx]

    return pd_prob, label
