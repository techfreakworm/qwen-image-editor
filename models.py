"""Model constants, device helpers, and dimension utilities for Qwen Image Editor."""

from __future__ import annotations

import math
import os

# ---------------------------------------------------------------------------
# Model & LoRA identifiers
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LORA_ADAPTER_NAME = "lightning"

# ---------------------------------------------------------------------------
# Lightning scheduler configuration
# Copied verbatim from diffusers-zerogpu-api.md §2.
# The Lightning distillation was trained with shift=3 (log(3)) and
# exponential time shifting — using the default scheduler gives degraded
# results when combined with the Lightning LoRA.
# ---------------------------------------------------------------------------
LIGHTNING_SCHEDULER_CONFIG: dict = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # shift=3 used during distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # same as base_shift for flat schedule
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def on_spaces() -> bool:
    """Return True iff running inside a Hugging Face ZeroGPU Space."""
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def auto_device() -> str:
    """Detect the best available compute device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_dimensions(image: object, max_pixels: int = 1024 * 1024, multiple: int = 16) -> tuple[int, int]:
    """Return (width, height) fitting within max_pixels, rounded down to `multiple`, min 256.

    Preserves aspect ratio as closely as the multiple constraint allows.
    Area is guaranteed <= max_pixels except in extreme aspect-ratio cases where
    the shorter dimension is clamped up to the 256 minimum.
    """
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        w = int(w * scale)
        h = int(h * scale)

    # Floor to nearest multiple
    w = (w // multiple) * multiple
    h = (h // multiple) * multiple

    # Enforce minimum of 256 on each side
    w = max(256, w)
    h = max(256, h)

    return w, h


def should_cpu_offload(device: str) -> bool:
    """Return True iff device is local CUDA with < 40 GB free VRAM.

    Always returns False on ZeroGPU Spaces (the runtime manages placement)
    and on mps/cpu (offload does not apply). Returns False gracefully when
    torch is not installed.
    """
    if device != "cuda" or on_spaces():
        return False
    try:
        import torch

        free_bytes, _total = torch.cuda.mem_get_info()
        return free_bytes / (1024**3) < 40.0
    except Exception:
        return False
