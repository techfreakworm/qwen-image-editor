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
    """ZeroGPU @spaces.GPU duration budget in seconds.

    The Space runs on the RTX Pro 6000 Blackwell ZeroGPU fleet, on the ``xlarge``
    (96 GB) tier because the ~58 GB model does not fit ``large`` (48 GB). xlarge
    DOUBLES the requested duration for ZeroGPU's per-call ceiling check. Live
    probing showed 290 s and 156 s requests both rejected ("ZeroGPU illegal
    duration"), so the per-call ceiling is ~120 s requested (ZeroGPU's documented
    max) — i.e. real duration must be ≲ 60 s on xlarge. The 58 GB model packs at
    startup and materializes to the GPU in only ~15-40 s (~6 GB/s, per the Space
    logs) + ~1.4 s/step, so Fast (4 steps) fits comfortably in this budget.

    Fast (4 steps) → 54 s (108 s requested, under the ceiling). Quality (more
    steps) needs ~80 s real (~160 s requested) which exceeds the ceiling, so it
    cannot fit on xlarge and would require fp8 (to reach the 1x ``large`` tier);
    it is capped here at 58 s and will abort until that lands.
    """
    steps = int(params.get("steps", 4))
    return min(58, 50 + steps)


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
