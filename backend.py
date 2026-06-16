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

    On the ``xlarge`` ZeroGPU tier (the ~58 GB model needs 96 GB) the request is
    DOUBLED for the per-call ceiling check. Confirmed live: duration=180 (360 s
    requested) is rejected ("ZeroGPU illegal duration"), while duration=54
    (108 s requested) is ACCEPTED but the task is "GPU task aborted" — the
    per-call execution (model materialization + inference) exceeds a 54 s budget.
    So we want the highest legal budget: 145 s (290 s requested, just under the
    360 s that was rejected). If even this aborts, the 58 GB bf16 model is too
    slow per-call for xlarge and needs fp8 (to reach the 1x ``large`` tier).

    Override via the ``QIE_GPU_DURATION`` Space variable to retune the budget
    without a code redeploy (a Space restart picks it up).
    """
    override = os.environ.get("QIE_GPU_DURATION")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    return 145


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
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        QwenImageEditPlusPipeline,
        QwenImageTransformer2DModel,
    )
    from optimum.quanto import freeze, qfloat8, quantize

    import models

    device = models.auto_device()

    # fp8 weight-only quantize the transformer (~40 GB bf16 -> ~20 GB fp8) with
    # optimum-quanto. Weights are stored fp8 but the matmul stays bf16, so bf16
    # LoRAs still apply via dynamic set_adapters (we never fuse). Text encoder +
    # VAE stay bf16 -> ~34 GB resident, which fits ZeroGPU's `large` (48 GB) tier
    # and ~halves the per-call GPU materialization, so the call completes inside
    # ZeroGPU's proxy-token TTL. (The bf16 58 GB model took ~140 s/call ->
    # exceeded the TTL -> the image was produced server-side but never delivered.)
    # quanto (not torchao) is used because diffusers@973a077 eagerly imports its
    # torchao quantizer, and no torchao version is compatible with both that
    # pinned diffusers and the Space's torch.
    transformer = QwenImageTransformer2DModel.from_pretrained(
        models.MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        models.MODEL_ID,
        transformer=transformer,
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
