"""Mode handlers — pure functions over QwenImageEditPlusPipeline + params dict."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image

import models


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


def _run(pipe: Any, params: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Run inference and return (output_image, metadata).

    Validates that at least one image is present, applies the speed mode,
    resolves image dimensions and seed, then calls the pipeline.
    """
    images: list[Any] = params["images"]
    if not images:
        raise ValueError("at least one image is required; params['images'] is empty")

    _apply_speed(pipe, params["speed"])

    w, h = models.fit_dimensions(images[0])

    seed_in = params["seed"]
    seed = random.randint(0, 2**32 - 1) if seed_in < 0 else int(seed_in)

    import torch  # deferred — CI has no torch installed

    # Normalize device: pipe.device may be a str or a torch.device object.
    # str(torch.device("cuda:0")) == "cuda:0", str("cpu") == "cpu" — safe in both cases.
    device = str(getattr(pipe, "device", "cpu"))
    gen = torch.Generator(device).manual_seed(seed)

    out = pipe(
        image=images,
        prompt=params["prompt"],
        negative_prompt=params["negative_prompt"],
        true_cfg_scale=params["true_cfg"],
        num_inference_steps=params["steps"],
        height=h,
        width=w,
        generator=gen,
    )

    meta: dict[str, Any] = {
        "mode": params["mode"],
        "speed": params["speed"],
        "steps": params["steps"],
        "true_cfg": params["true_cfg"],
        "seed": seed,
        "width": w,
        "height": h,
        "num_inputs": len(images),
    }
    return out.images[0], meta


def call_edit(pipe: Any, params: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Edit mode: single input image + instruction -> edited image.

    Expects params["images"] == [target_image].
    """
    p = dict(params)
    p["mode"] = "edit"
    return _run(pipe, p)


def call_compose(pipe: Any, params: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Compose mode: target image + up to 2 optional reference images -> composed edit.

    None slots in params["images"] are dropped before the pipeline call so the
    pipeline always receives a contiguous list of 1..3 real images.
    """
    p = dict(params)
    p["mode"] = "compose"
    p["images"] = [img for img in params["images"] if img is not None]
    return _run(pipe, p)


DISPATCH: dict[str, Any] = {
    "edit": call_edit,
    "compose": call_compose,
}
