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

# ====== FFmpeg bundled with backend (no PATH needed) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = os.path.join(BASE_DIR, "ffmpeg", "ffmpeg.exe")
FFPROBE_PATH = os.path.join(BASE_DIR, "ffmpeg", "ffprobe.exe")

# Check they exist and configure pydub
if not os.path.exists(FFMPEG_PATH) or not os.path.exists(FFPROBE_PATH):
    # This will show clearly in the logs instead of WinError 2
    print("[ERROR] FFmpeg not found at:")
    print("  ", FFMPEG_PATH)
    print("  ", FFMPEG_PATH)
    print("[HINT] Put ffmpeg.exe and ffprobe.exe in a folder called 'ffmpeg' "
          "inside your backend directory (sixtyscan-ai-backend/ffmpeg/).")
else:
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg = FFMPEG_PATH
    AudioSegment.ffprobe = FFPROBE_PATH
    print("[INFO] Using bundled FFmpeg:", FFMPEG_PATH)
# ==========================================================


def audio_bytes_to_mel_image(audio_bytes: bytes) -> Image.Image:
    """
    Accept ANY audio format (webm, wav, mp3, m4a, etc.).
    Convert to WAV using pydub + ffmpeg.
    Then produce a mel spectrogram as a PIL Image.
    """
    # Extra safety: if ffmpeg is missing, fail with a clear message
    if not os.path.exists(FFMPEG_PATH) or not os.path.exists(FFPROBE_PATH):
        raise RuntimeError(
            "FFmpeg executables not found.\n"
            "Expected at:\n"
            f"  {FFMPEG_PATH}\n"
            f"  {FFPROBE_PATH}\n"
            "Make sure ffmpeg.exe and ffprobe.exe are inside the 'ffmpeg' "
            "folder in your backend directory."
        )

    # 1) decode audio bytes with pydub (FFmpeg under the hood)
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
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
