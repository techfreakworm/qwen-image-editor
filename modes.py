"""Mode handlers — pure functions over QwenImageEditPlusPipeline + params dict."""

from __future__ import annotations

import random
import threading
from typing import Any

from PIL import Image

import models

# Serialize MPS inference: one GPU, one shared pipeline whose scheduler/adapter are
# swapped per request. Two concurrent runs would race that state + corrupt the
# memory peak measurement + race the calibration cache. (The app also caps queue
# concurrency to 1; this is correct-by-construction belt-and-suspenders.)
_GPU_LOCK = threading.Lock()


def _is_gpu_oom(e: BaseException) -> bool:
    """True for any GPU allocation-failure RuntimeError (MPS or CUDA).

    Covers the raw allocator OOM ("MPS backend out of memory" / "CUDA out of memory"),
    the MPS high-watermark throw (a DIFFERENT message), and CUDA OOM variants — otherwise
    those would escape the OOM-retry net and crash, defeating the never-OOM guarantee.
    """
    s = str(e).lower()
    return any(
        k in s
        for k in (
            "out of memory", "watermark", "mps allocated", "cannot allocate",
            "insufficient memory", "cuda error", "cublas",
        )
    )


def _gpu_empty_cache(torch: Any, dt: str) -> None:
    if dt == "mps":
        torch.mps.empty_cache()
    elif dt == "cuda":
        torch.cuda.empty_cache()


def _gpu_synchronize(torch: Any, dt: str) -> None:
    if dt == "mps":
        torch.mps.synchronize()
    elif dt == "cuda":
        torch.cuda.synchronize()


def _gpu_allocated_gb(torch: Any, dt: str) -> float:
    """Currently-allocated GPU memory (GB) — the resident baseline before inference."""
    if dt == "mps":
        return torch.mps.driver_allocated_memory() / (1024**3)
    if dt == "cuda":
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def _gpu_peak_gb(torch: Any, dt: str, baseline_gb: float) -> float:
    """Activation peak (GB) above the resident baseline, for calibration."""
    if dt == "mps":
        return max(0.0, torch.mps.driver_allocated_memory() / (1024**3) - baseline_gb)
    if dt == "cuda":
        return max(0.0, torch.cuda.max_memory_allocated() / (1024**3) - baseline_gb)
    return 0.0


def _apply_speed(pipe: Any, speed: str) -> None:
    """Configure the pipeline for Fast (Lightning LoRA) or Quality (default) mode.

    Fast:    swap in the Lightning scheduler + enable the Lightning LoRA adapter.
    Quality: restore the default scheduler + disable the LoRA adapter entirely.
    Both operations are cheap, reversible, and ZeroGPU-safe (no weight fusion/unfusion).
    """
    if speed == "Fast":
        pipe.scheduler = pipe._qie_lightning_scheduler
        pipe.set_adapters([models.LORA_ADAPTER_NAME], [1.0])
    else:
        pipe.scheduler = pipe._qie_default_scheduler
        pipe.disable_lora()


def _step_callback(progress: Any, total_steps: int) -> Any:
    """Build a diffusers ``callback_on_step_end`` that drives a ``gr.Progress`` with clear
    labels ("Generating — step N/M"). Returns None when there's no progress object (CI /
    non-UI calls), so the pipeline call is unchanged off the UI path.

    Using an explicit step callback (instead of Gradio's ``track_tqdm``) avoids the
    confusing "Downloading (incomplete total…) 0/0 B" placeholder Gradio renders for the
    pipeline's internal, total-less tqdm bars during the input-encode phase.
    """
    if progress is None:
        return None

    def _cb(_pipe: Any, step: int, _timestep: Any, callback_kwargs: dict) -> dict:
        progress((step + 1) / max(1, total_steps), desc=f"Generating — step {step + 1}/{total_steps}")
        return callback_kwargs

    return _cb


def _run(pipe: Any, params: dict[str, Any], progress: Any = None) -> tuple[Image.Image, dict[str, Any]]:
    """Run inference and return (output_image, metadata).

    Validates that at least one image is present, applies the speed mode, resolves
    dimensions and seed, then calls the pipeline. On MPS an OOM preflight runs first:
    it sizes the request against a live memory budget and auto-degrades down a ladder
    (resolution -> refs -> drop CFG double-pass -> Quality->Fast) so inference never
    OOMs; the math + any degradation are surfaced in meta. The CUDA/CPU path is
    unchanged. ``progress`` (optional gr.Progress) drives a clean step bar.
    """
    images: list[Any] = params["images"]
    if not images:
        raise ValueError("at least one image is required; params['images'] is empty")

    import torch  # deferred — CI has no torch installed

    # Normalize device: pipe.device may be a str or a torch.device object.
    device = str(getattr(pipe, "device", "cpu"))
    device_type = device.split(":")[0]
    # ZeroGPU quirk: after the spaces pack/restore, pipe.device can read "cpu" at the
    # START of the call even though execution happens on a real CUDA GPU inside the
    # @spaces.GPU fork. Trust the Space environment over the (stale) device property so
    # the CUDA activation preflight + never-OOM degrade path actually run on ZeroGPU
    # (otherwise the simple/no-budget path is taken and a heavy request can OOM-SIGKILL).
    if device_type not in ("mps", "cuda") and models.on_spaces():
        device_type = "cuda"
    is_gpu = device_type in ("mps", "cuda")

    mode = params["mode"]
    speed = params["speed"]
    steps = int(params["steps"])
    true_cfg = float(params["true_cfg"])
    prompt = params["prompt"]
    negative_prompt = params["negative_prompt"]

    seed_in = params["seed"]
    seed = random.randint(0, 2**32 - 1) if seed_in < 0 else int(seed_in)

    # --- CPU / mocked path (no GPU memory budgeting needed) -----------------------------
    if not is_gpu:
        _apply_speed(pipe, speed)
        w, h = models.fit_dimensions(images[0])
        gen = torch.Generator(device).manual_seed(seed)
        if progress is not None:
            progress(0.0, desc="Encoding inputs…")
        out = pipe(
            image=images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg,
            num_inference_steps=steps,
            height=h,
            width=w,
            generator=gen,
            callback_on_step_end=_step_callback(progress, steps),
        )
        meta = {
            "mode": mode, "speed": speed, "steps": steps, "true_cfg": true_cfg,
            "seed": seed, "width": w, "height": h, "num_inputs": len(images),
        }
        return out.images[0], meta

    # --- GPU path (MPS or CUDA/ZeroGPU): serialized, OOM-preflight + reactive degrade ----
    # Same activation-centric never-OOM logic on both: the ~58 GB model is already resident,
    # so only the activation must fit the free budget (MPS: free unified RAM capped by the
    # working set; CUDA/ZeroGPU: free VRAM via torch.cuda.mem_get_info). On ZeroGPU xlarge
    # (96 GB) Edit fits but heavy Compose can exceed -> auto-degrade (the user's mandate).
    import gc

    import memory

    with _GPU_LOCK:
        gc.collect()
        _gpu_empty_cache(torch, device_type)
        base_w, base_h = images[0].size
        n_ref0 = max(0, len(images) - 1)
        plan = memory.plan_request(device_type, mode, base_w, base_h, n_ref0, speed, steps, true_cfg)
        if plan["refused"]:
            raise RuntimeError(plan["note"])

        w, h = plan["width"], plan["height"]
        steps, true_cfg, speed, n_ref = plan["steps"], plan["true_cfg"], plan["speed"], plan["n_ref"]
        degrades = list(plan["degrades"])

        out = None
        last_err: Exception | None = None
        for _attempt in range(8):  # bounded reactive degrade — hard never-OOM guarantee
            imgs = images[: n_ref + 1]
            _apply_speed(pipe, speed)
            gen = torch.Generator("cpu").manual_seed(seed)  # CPU generator is safe on MPS + CUDA
            baseline_gb = _gpu_allocated_gb(torch, device_type)
            if device_type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            if progress is not None:
                progress(0.0, desc="Encoding inputs…")
            try:
                out = pipe(
                    image=imgs,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    true_cfg_scale=true_cfg,
                    num_inference_steps=steps,
                    height=h,
                    width=w,
                    generator=gen,
                    callback_on_step_end=_step_callback(progress, steps),
                )
                break
            except RuntimeError as e:
                if not _is_gpu_oom(e):
                    raise
                last_err = e
                del gen
                gc.collect()
                _gpu_synchronize(torch, device_type)
                _gpu_empty_cache(torch, device_type)
                # Self-correct AFTER cleanup: at the OOM instant the heap is at its sticky
                # peak, so the budget would read ~0 -> no-op penalty. A clean heap returns the
                # real budget the config overflowed, so the next request degrades preemptively.
                memory.penalize(device_type, h, w, n_ref, true_cfg > 1.0, memory.activation_budget_gb(device_type))
                nxt = memory.step_down(w, h, n_ref, speed, true_cfg, steps, base_w, base_h, mode)
                if nxt is None:
                    raise RuntimeError(f"GPU OOM at the smallest config and cannot degrade further: {e}") from e
                w, h, n_ref = nxt["width"], nxt["height"], nxt["n_ref"]
                speed, true_cfg, steps = nxt["speed"], nxt["true_cfg"], nxt["steps"]
                degrades.append(f"OOM-retry->{w}x{h} {speed} {n_ref + 1}img")
        if out is None:  # pragma: no cover - defensive
            raise RuntimeError(f"GPU inference failed after retries: {last_err}")

        peak_gb = _gpu_peak_gb(torch, device_type, baseline_gb)
        memory.record_peak(device_type, mode, h, w, n_ref, true_cfg > 1.0, peak_gb)

        meta = {
            "mode": mode, "speed": speed, "steps": steps, "true_cfg": true_cfg,
            "seed": seed, "width": w, "height": h, "num_inputs": n_ref + 1,
            "preflight": plan["note"], "budget_gb": plan["budget_gb"],
            "need_gb": plan["need_gb"], "measured_peak_gb": round(peak_gb, 1),
        }
        if degrades:
            meta["degrades"] = degrades

        gc.collect()
        _gpu_empty_cache(torch, device_type)
        return out.images[0], meta


def call_edit(pipe: Any, params: dict[str, Any], progress: Any = None) -> tuple[Image.Image, dict[str, Any]]:
    """Edit mode: single input image + instruction -> edited image.

    Expects params["images"] == [target_image].
    """
    p = dict(params)
    p["mode"] = "edit"
    return _run(pipe, p, progress)


def call_compose(pipe: Any, params: dict[str, Any], progress: Any = None) -> tuple[Image.Image, dict[str, Any]]:
    """Compose mode: target image + up to 2 optional reference images -> composed edit.

    None slots in params["images"] are dropped before the pipeline call so the
    pipeline always receives a contiguous list of 1..3 real images.
    """
    p = dict(params)
    p["mode"] = "compose"
    p["images"] = [img for img in params["images"] if img is not None]
    return _run(pipe, p, progress)


DISPATCH: dict[str, Any] = {
    "edit": call_edit,
    "compose": call_compose,
}
