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


def _upcast_vae_for_mps(pipe: Any, torch: Any) -> None:
    """Opt-in fp32 VAE for MPS that still accepts the pipeline's bf16 tensors.

    The QwenImageEditPlus pipeline calls ``vae.encode``/``vae.decode`` without casting,
    so a bare fp32 VAE mismatches the bf16 inputs. We move the VAE to fp32 and wrap its
    encode/decode to upcast inputs and downcast the decoded sample back to the pipeline
    dtype — keeping the rest of the pipeline in bf16. Only used when QIE_MPS_VAE_FP32=1
    (i.e. if a bf16 VAE is ever found to produce black/NaN images on this hardware).
    """
    vae = pipe.vae
    pipe_dtype = getattr(pipe, "dtype", torch.bfloat16)
    vae.to(torch.float32)
    if getattr(vae, "config", None) is not None:
        vae.config.force_upcast = True
    _orig_encode, _orig_decode = vae.encode, vae.decode

    def _encode(x, *a, **k):
        return _orig_encode(x.to(torch.float32), *a, **k)

    def _decode(z, *a, **k):
        out = _orig_decode(z.to(torch.float32), *a, **k)
        if hasattr(out, "sample"):
            out.sample = out.sample.to(pipe_dtype)
        return out

    vae.encode, vae.decode = _encode, _decode


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

    device = models.auto_device()

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

    # fp8 weight-only quantization is a CUDA-only path (torchao). It is used ONLY on
    # CUDA (local GPU or ZeroGPU), where shrinking the transformer ~40 GB bf16 -> ~20 GB
    # fp8 matters to fit the `large` (48 GB) tier and ~halve the per-call materialization
    # so the ZeroGPU call completes inside the proxy-token TTL. On Apple Silicon (MPS) or
    # CPU there is no torchao fp8 kernel, so we load the model full-precision bf16: this
    # machine's 128 GB unified memory holds the ~58 GB bf16 model comfortably (transformer
    # 40.9 GB + text_encoder 16.6 GB + vae 0.25 GB). The fp8 import is deferred into this
    # branch so MPS/CPU runs never require torchao to be installed.
    use_fp8 = device == "cuda"

    if use_fp8:
        from diffusers import TorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig

        transformer = QwenImageTransformer2DModel.from_pretrained(
            models.MODEL_ID,
            subfolder="transformer",
            quantization_config=TorchAoConfig(Float8WeightOnlyConfig()),
            torch_dtype=torch.bfloat16,
        )
    else:
        # MPS / CPU: full bf16, no quantization (torchao fp8 is CUDA-only).
        transformer = QwenImageTransformer2DModel.from_pretrained(
            models.MODEL_ID,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

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
    # NOTE: on Apple Silicon's UNIFIED memory, enable_model_cpu_offload saves no
    # physical RAM (CPU+GPU share one 128 GB pool), so MPS goes straight to .to(mps)
    # (models.should_cpu_offload already returns False for mps).
    if models.on_spaces():
        pipe.to("cuda")
    elif models.should_cpu_offload(device):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # MPS-specific setup, applied AFTER .to("mps"):
    if device == "mps":
        # VAE dtype: keep bf16 to match the pipeline. The QwenImageEditPlus pipeline
        # calls vae.encode(image) on a bf16-preprocessed image WITHOUT upcasting, so
        # forcing the VAE to fp32 breaks encode ("Input bf16 vs bias float"). If a bf16
        # VAE decode ever yields black/NaN images on MPS, set QIE_MPS_VAE_FP32=1 to opt
        # into the fp32 VAE + input-upcast path (_upcast_vae_for_mps).
        if os.environ.get("QIE_MPS_VAE_FP32", "0") == "1":
            _upcast_vae_for_mps(pipe, torch)
        # VAE tiling + slicing cap the decode memory spike (nearly free).
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
