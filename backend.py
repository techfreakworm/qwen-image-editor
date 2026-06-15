"""QwenImageEditBackend — wraps the diffusers pipeline; applies @spaces.GPU on HF Spaces."""

from __future__ import annotations

import os
from typing import Any

# Spaces import is optional — running locally we don't have it.
try:
    import spaces  # type: ignore
except ImportError:
    spaces = None  # type: ignore[assignment]

import modes

_ON_SPACES = bool(os.environ.get("SPACES_ZERO_GPU"))


def _identity(fn):
    return fn


def duration_for(mode: str, params: dict[str, Any]) -> int:
    """ZeroGPU slot budget in seconds (returns the 180 s ceiling).

    Measured on this Space: the per-call cost is dominated by the ~58 GB model's
    GPU materialization (~130 s); each denoising step is only ~1.4 s. So both
    presets need the full window the official Qwen-Image-Edit-2511 Space relies on.
    Quality's step count is kept low enough (see app._speed_defaults) to finish
    within this budget.
    """
    return 180


def _duration_arg(*args: Any, **kwargs: Any) -> int:
    """Locate the params dict among the decorated method's call args.

    spaces passes generate()'s args (which include ``self``) to this callable, so
    we scan for the params dict rather than relying on a fixed positional index.
    """
    params = next((a for a in args if isinstance(a, dict)), {})
    return duration_for("", params)


_GPU = spaces.GPU(duration=_duration_arg) if (spaces is not None and _ON_SPACES) else _identity


def _build_pipeline() -> Any:
    """Build and configure a QwenImageEditPlusPipeline.

    Heavy imports (torch, diffusers) are deferred here so that importing
    backend.py in CI (no torch installed) is always safe.

    Pipeline is pre-loaded with the Lightning LoRA adapter and starts in
    Fast mode (Lightning scheduler + adapter enabled at weight 1.0).
    Switching to Quality mode is handled by modes._apply_speed per request.
    """
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

    import models

    device = models.auto_device()

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        models.MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # Stash the bundled default scheduler for Quality mode.
    pipe._qie_default_scheduler = pipe.scheduler

    # Build and stash the Lightning scheduler for Fast mode.
    pipe._qie_lightning_scheduler = FlowMatchEulerDiscreteScheduler.from_config(models.LIGHTNING_SCHEDULER_CONFIG)

    # Load (but do NOT fuse) the Lightning LoRA so we can toggle it per-request.
    pipe.load_lora_weights(
        models.LORA_REPO,
        weight_name=models.LORA_FILE,
        adapter_name=models.LORA_ADAPTER_NAME,
    )

    # Device placement — on_spaces takes priority; cpu-offload for small local GPUs.
    if models.on_spaces():
        pipe.to("cuda")
    elif models.should_cpu_offload(device):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # Start in Fast state: Lightning scheduler + LoRA adapter at full weight.
    pipe.scheduler = pipe._qie_lightning_scheduler
    pipe.set_adapters([models.LORA_ADAPTER_NAME], [1.0])

    return pipe


class QwenImageEditBackend:
    """One-process backend wrapping the QwenImageEditPlusPipeline."""

    def __init__(self) -> None:
        self.pipeline = _build_pipeline()

    @_GPU
    def generate(self, mode: str, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Route the request to the appropriate mode handler.

        Raises ValueError for unrecognised modes so callers get a clear error
        rather than an opaque AttributeError / KeyError.
        """
        handler = modes.DISPATCH.get(mode)
        if handler is None:
            raise ValueError(f"unknown mode: {mode!r}; expected one of {list(modes.DISPATCH)}")
        return handler(self.pipeline, params)


def generate_with_retry(
    backend_instance: QwenImageEditBackend,
    mode: str,
    params: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Call backend_instance.generate; on ZeroGPU GPU-task-abort, retry exactly once.

    Any other exception propagates immediately without a retry.
    """
    try:
        return backend_instance.generate(mode, params)
    except Exception as e:
        if "gpu task aborted" in str(e).lower():
            return backend_instance.generate(mode, params)
        raise
