"""Tests for models.py — constants, env helpers, fit_dimensions, scheduler config."""

from __future__ import annotations

import importlib.util
import math

import pytest
from PIL import Image

import models

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_model_id():
    assert models.MODEL_ID == "Qwen/Qwen-Image-Edit-2511"


def test_lora_repo():
    assert models.LORA_REPO == "lightx2v/Qwen-Image-Edit-2511-Lightning"


def test_lora_file():
    assert models.LORA_FILE == "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


def test_lora_adapter_name():
    assert models.LORA_ADAPTER_NAME == "lightning"


# ---------------------------------------------------------------------------
# LIGHTNING_SCHEDULER_CONFIG
# ---------------------------------------------------------------------------


def test_scheduler_config_time_shift_type():
    assert models.LIGHTNING_SCHEDULER_CONFIG["time_shift_type"] == "exponential"


def test_scheduler_config_base_shift():
    assert models.LIGHTNING_SCHEDULER_CONFIG["base_shift"] == math.log(3)


def test_scheduler_config_max_shift():
    assert models.LIGHTNING_SCHEDULER_CONFIG["max_shift"] == math.log(3)


def test_scheduler_config_use_dynamic_shifting():
    assert models.LIGHTNING_SCHEDULER_CONFIG["use_dynamic_shifting"] is True


def test_scheduler_config_has_required_keys():
    required = {
        "base_image_seq_len",
        "base_shift",
        "invert_sigmas",
        "max_image_seq_len",
        "max_shift",
        "num_train_timesteps",
        "shift",
        "shift_terminal",
        "stochastic_sampling",
        "time_shift_type",
        "use_beta_sigmas",
        "use_dynamic_shifting",
        "use_exponential_sigmas",
        "use_karras_sigmas",
    }
    assert required.issubset(models.LIGHTNING_SCHEDULER_CONFIG.keys())


# ---------------------------------------------------------------------------
# on_spaces
# ---------------------------------------------------------------------------


def test_on_spaces_false_when_env_absent(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    assert models.on_spaces() is False


def test_on_spaces_true_when_env_set(monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    assert models.on_spaces() is True


def test_on_spaces_true_for_nonempty_string(monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    assert models.on_spaces() is True


def test_on_spaces_false_for_empty_string(monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "")
    assert models.on_spaces() is False


# ---------------------------------------------------------------------------
# fit_dimensions helpers
# ---------------------------------------------------------------------------


def _img(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h))


# --- multiples of 16 ---


def test_fit_dimensions_landscape_multiple_of_16():
    w, h = models.fit_dimensions(_img(1920, 1080))
    assert w % 16 == 0
    assert h % 16 == 0


def test_fit_dimensions_portrait_multiple_of_16():
    w, h = models.fit_dimensions(_img(1080, 1920))
    assert w % 16 == 0
    assert h % 16 == 0


def test_fit_dimensions_square_multiple_of_16():
    w, h = models.fit_dimensions(_img(2000, 2000))
    assert w % 16 == 0
    assert h % 16 == 0


# --- area cap ---


def test_fit_dimensions_area_cap_landscape():
    w, h = models.fit_dimensions(_img(1920, 1080))
    assert w * h <= 1024 * 1024


def test_fit_dimensions_area_cap_portrait():
    w, h = models.fit_dimensions(_img(1080, 1920))
    assert w * h <= 1024 * 1024


def test_fit_dimensions_area_cap_large_square():
    w, h = models.fit_dimensions(_img(2048, 2048))
    assert w * h <= 1024 * 1024


def test_fit_dimensions_custom_max_pixels():
    w, h = models.fit_dimensions(_img(1024, 1024), max_pixels=512 * 512)
    assert w * h <= 512 * 512


# --- minimum 256 ---


def test_fit_dimensions_min_256_tiny_image():
    w, h = models.fit_dimensions(_img(32, 32))
    assert w >= 256
    assert h >= 256


def test_fit_dimensions_min_256_small_image():
    w, h = models.fit_dimensions(_img(100, 100))
    assert w >= 256
    assert h >= 256


def test_fit_dimensions_min_256_thin_portrait():
    # Thin portrait: short side would be tiny without min clamp
    w, h = models.fit_dimensions(_img(64, 512))
    assert w >= 256
    assert h >= 256


# --- aspect ratio within tolerance ---


def test_fit_dimensions_aspect_landscape_within_5pct():
    original_ratio = 1920 / 1080
    w, h = models.fit_dimensions(_img(1920, 1080))
    new_ratio = w / h
    assert abs(new_ratio - original_ratio) / original_ratio < 0.05


def test_fit_dimensions_aspect_portrait_within_5pct():
    original_ratio = 1080 / 1920
    w, h = models.fit_dimensions(_img(1080, 1920))
    new_ratio = w / h
    assert abs(new_ratio - original_ratio) / original_ratio < 0.05


def test_fit_dimensions_aspect_wide_within_5pct():
    # 16:9 wide variant
    original_ratio = 1280 / 720
    w, h = models.fit_dimensions(_img(1280, 720))
    new_ratio = w / h
    assert abs(new_ratio - original_ratio) / original_ratio < 0.05


# --- no-op when image already fits ---


def test_fit_dimensions_no_change_when_already_fits():
    # 768x1024 = 786432 < 1048576, already multiples of 16, both >= 256
    w, h = models.fit_dimensions(_img(768, 1024))
    assert w == 768
    assert h == 1024


def test_fit_dimensions_exact_max_pixels():
    # Exactly 1024x1024 = max_pixels — no scaling, no rounding needed
    w, h = models.fit_dimensions(_img(1024, 1024))
    assert w == 1024
    assert h == 1024


# --- custom multiple ---


def test_fit_dimensions_custom_multiple():
    w, h = models.fit_dimensions(_img(1920, 1080), multiple=32)
    assert w % 32 == 0
    assert h % 32 == 0


# ---------------------------------------------------------------------------
# auto_device — requires torch; skip gracefully when absent
# ---------------------------------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_auto_device_returns_valid_device():
    device = models.auto_device()
    assert device in ("cuda", "mps", "cpu")


# ---------------------------------------------------------------------------
# should_cpu_offload — basic logic tests (no real GPU needed)
# ---------------------------------------------------------------------------


def test_should_cpu_offload_false_on_spaces(monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    assert models.should_cpu_offload("cuda") is False


def test_should_cpu_offload_false_for_mps(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    assert models.should_cpu_offload("mps") is False


def test_should_cpu_offload_false_for_cpu(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    assert models.should_cpu_offload("cpu") is False


def test_should_cpu_offload_false_when_torch_unavailable(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    # Simulate torch raising on cuda check by patching the function's import
    import sys

    original = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        result = models.should_cpu_offload("cuda")
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original
    assert result is False
