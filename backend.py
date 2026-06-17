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


# ZeroGPU GPU size: "xlarge" (96 GB, full RTX Pro 6000 Blackwell, 2x quota) is required
# because the ~58 GB bf16 model does NOT fit the default "large" (48 GB) tier. xlarge keeps
# the whole model resident with NO cpu_offload, eliminating the slow per-call materialization
# that exceeded the proxy-TTL on the 48 GB tier (the historical "image produced but never
# delivered" failure). Override via QIE_GPU_SIZE.
_GPU_SIZE = os.environ.get("QIE_GPU_SIZE", "xlarge")
_GPU = spaces.GPU(duration=_duration_arg, size=_GPU_SIZE) if (spaces is not None and _ON_SPACES) else _identity


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

    import models

    # On ZeroGPU the build runs under CPU-emulation where auto_device() can misreport
    # (cuda.is_available() may be False at module scope), yet the real per-call device is
    # CUDA. Trust the Space env so the dtype/VAE/placement decisions below are consistent
    # with how the model actually runs. Off-Spaces, autodetect (cuda > mps > cpu).
    device = "cuda" if models.on_spaces() else models.auto_device()

    # LOAD gate (MPS): confirm the ~58 GB resident weights fit BEFORE materializing
    # shards, so a low-memory box refuses cleanly instead of OOM / swap-thrashing
    # mid-load. A resident-weights overflow cannot be fixed by lowering resolution.
    if device == "mps":
        import memory

        ok, need, budget = memory.load_gate(device)
        if not ok:
            raise RuntimeError(
                f"Insufficient memory to load Qwen-Image-Edit-2511 on MPS: need ~{need:.0f} GB, "
                f"budget ~{budget:.0f} GB (available {memory.available_gb():.0f} GB - reserve). "
                f"Free ~{need - budget:.0f} GB then retry. Top consumers: {memory.top_consumers()}"
            )

    # Precision: default to full bf16 everywhere (the ~58 GB model — transformer 40.9 +
    # text_encoder 16.6 + vae 0.25 — fits the MPS 128 GB unified pool AND the ZeroGPU
    # `xlarge` 96 GB tier with the whole model resident, no offload). fp8 weight-only
    # quantization (torchao, CUDA-only) is now OPT-IN via QIE_USE_FP8=1: it shrinks the
    # transformer ~40->~20 GB to fit the smaller `large` 48 GB tier, but the torchao/
    # diffusers fp8 stack proved unstable, so bf16 on xlarge is the default/working path.
    use_fp8 = device == "cuda" and os.environ.get("QIE_USE_FP8", "0") == "1"

    if use_fp8:
        from diffusers import TorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig

        transformer = QwenImageTransformer2DModel.from_pretrained(
            models.MODEL_ID,
            subfolder="transformer",
            quantization_config=TorchAoConfig(Float8WeightOnlyConfig()),
            torch_dtype=torch.bfloat16,
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            models.MODEL_ID,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
    else:
        # Default bf16: load the WHOLE pipeline directly (no separately-loaded transformer).
        # On ZeroGPU a separately-constructed component is a suspected disruptor of the
        # snapshot/restore CUDA placement (observed: all components landed on CPU in the
        # fork). Direct load mirrors the official Qwen Space and lets `spaces` track every
        # component for the cuda snapshot.
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
    # NOTE: on Apple Silicon's UNIFIED memory, enable_model_cpu_offload saves no
    # physical RAM (CPU+GPU share one 128 GB pool), so MPS goes straight to .to(mps)
    # (models.should_cpu_offload already returns False for mps).
    if models.on_spaces():
        # ZeroGPU: leave the model on CPU at module level and move it to cuda INSIDE the
        # @spaces.GPU fork (generate()). A module-level pipe.to("cuda") is snapshotted, and on
        # per-call restore the params land back on CPU while a ~36 GB orphaned cuda copy stays
        # resident → the in-fork .to then makes a 2nd copy → ~94 GB duplication (no room for
        # activation). Building on CPU = no cuda orphan → one clean ~58 GB copy in the fork.
        pass
    elif models.should_cpu_offload(device):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # VAE tiling + slicing cap the decode memory spike — enabled on MPS (unified memory is
    # tight). Not needed on CUDA/ZeroGPU: the 96 GB xlarge tier decodes 1024^2 untiled with
    # ample headroom (measured activation peak ~4 GB).
    if device == "mps":
        for _fn in ("enable_vae_tiling", "enable_vae_slicing"):
            if hasattr(pipe, _fn):
                getattr(pipe, _fn)()

    # Start in Fast state: Lightning scheduler + LoRA adapter at full weight.
    pipe.scheduler = pipe._qie_lightning_scheduler
    pipe.set_adapters([models.LORA_ADAPTER_NAME], [1.0])

    return pipe


class QwenImageEditBackend:
    """One-process backend wrapping the QwenImageEditPlusPipeline."""

    def __init__(self) -> None:
        self.pipeline = _build_pipeline()

    @_GPU
    def generate(self, mode: str, params: dict[str, Any], progress: Any = None) -> tuple[Any, dict[str, Any]]:
        """Route the request to the appropriate mode handler.

        Raises ValueError for unrecognised modes so callers get a clear error
        rather than an opaque AttributeError / KeyError. ``progress`` (optional
        gr.Progress) is forwarded to the handler to drive a clean step bar.
        """
        # ZeroGPU placement: the model is built on CPU (a module-level .to("cuda") gets
        # snapshotted and restores the params back to CPU while leaving a duplicate cuda copy
        # resident). Moving to cuda HERE — inside the @spaces.GPU fork, where a real GPU is
        # attached — gives one clean ~55 GB resident copy with ~37 GB free for activation.
        # Idempotent; Spaces-only (local MPS/CPU placement is set at build time).
        if _ON_SPACES:
            self.pipeline.to("cuda")

        handler = modes.DISPATCH.get(mode)
        if handler is None:
            raise ValueError(f"unknown mode: {mode!r}; expected one of {list(modes.DISPATCH)}")
        return handler(self.pipeline, params, progress)


def generate_with_retry(
    backend_instance: QwenImageEditBackend,
    mode: str,
    params: dict[str, Any],
    progress: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Call backend_instance.generate; on ZeroGPU GPU-task-abort, retry exactly once.

    Any other exception propagates immediately without a retry.
    """
    try:
        return backend_instance.generate(mode, params, progress)
    except Exception as e:
        if "gpu task aborted" in str(e).lower():
            return backend_instance.generate(mode, params, progress)
        raise
