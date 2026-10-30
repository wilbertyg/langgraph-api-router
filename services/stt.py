import gc
import os
import librosa
import torch
from transformers import pipeline

MODEL_NAME = os.getenv("STT_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
# Parakeet handles ~6.5 min in one pass. Longer audio is cut into back-to-back windows
# with no overlap: the pipeline's overlap merging duplicates text for TDT models.
WINDOW_S = 300


def transcribe(audio_path: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Load pipeline into VRAM
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
        dtype=dtype,
    )

    try:
        # 2. Transcribe
        chunking = {}
        if librosa.get_duration(path=audio_path) > WINDOW_S:
            chunking = {"chunk_length_s": WINDOW_S, "stride_length_s": 0}
        result = pipe(audio_path, **chunking)
        return result.get("text", "") if isinstance(result, dict) else str(result)
    finally:
        # 3. Immediately dump from VRAM
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
