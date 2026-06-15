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
    """Estimate ZeroGPU duration for a request. Pure function; clamped to [60, 180].

    Fast mode always returns 60 (Lightning LoRA runs in ~10-15 s on H200).
    Quality mode scales linearly with the number of inference steps,
    clamped to the [60, 180] second range.
    """
    if params.get("speed") == "Fast":
        return 60
    steps = int(params.get("steps", 40))
    return int(min(180, max(60, 30 + steps * 3.5)))


_GPU = spaces.GPU(duration=lambda *a, **kw: duration_for(*a[1:3])) if (spaces is not None and _ON_SPACES) else _identity


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
