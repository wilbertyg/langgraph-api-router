import gc
import os
import torch
from transformers import pipeline

MODEL_NAME = os.getenv("STT_MODEL", "nvidia/parakeet-tdt-0.6b-v3")


def transcribe(audio_path: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Load pipeline into VRAM
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
        torch_dtype=torch_dtype,
    )

    try:
        # 2. Transcribe
        result = pipe(audio_path, chunk_length_s=30, batch_size=2)
        return result.get("text", "") if isinstance(result, dict) else str(result)
    finally:
        # 3. Immediately dump from VRAM
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
