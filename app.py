"""Qwen Image Editor — Gradio entrypoint."""

from __future__ import annotations

import os

# Apple Silicon: let PyTorch fall back to CPU for the small set of ops MPS
# doesn't implement. Must be set before any torch-touching import path is taken.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# MPS allocator watermarks (relative to recommended_max ~107.5 GB; ignored off-MPS).
# These MUST be set before the first torch MPS touch — hence the very top of app.py,
# before any torch-importing module. HIGH=1.0 (~107.5 GB) is the hard ceiling that turns
# a memory overflow into a catchable RuntimeError (the OOM-retry in modes._run degrades
# from it) instead of a swap-thrash hang. It sits a ~5 GB gap ABOVE the preflight budget
# cap (0.95*recommended ~102 GB) so allocator fragmentation on a legitimately-approved run
# doesn't spuriously throw. LOW=0.9 (~96.75 GB) is the cache-trim trigger, above the
# common-mode peak so normal runs don't churn. (LOW must be < HIGH or torch rejects it.)
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")

import gradio as gr

import backend
import models
import theme
import tooltips  # noqa: F401 — imported for public API availability
import ui

# ----- Lazy backend singleton ------------------------------------------------

_BACKEND: backend.QwenImageEditBackend | None = None


def _get_backend() -> backend.QwenImageEditBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = backend.QwenImageEditBackend()
    return _BACKEND


# ----- Speed preset helpers --------------------------------------------------


def _speed_defaults(speed: str) -> tuple:
    """Return (steps, true_cfg, gr.update(visible=...)) for a given speed preset.

    Fast:    4 steps, cfg=1.0, quality_grp hidden.
    Quality: 40 steps, cfg=4.0, quality_grp visible.
    """
    if speed == "Fast":
        return (4, 1.0, gr.update(visible=False))
    return (28, 4.0, gr.update(visible=True))


# ----- Generation event handlers ---------------------------------------------


def on_edit_generate(
    image,
    prompt,
    speed,
    steps,
    true_cfg,
    negative_prompt,
    seed,
    progress=gr.Progress(),  # noqa: B008 — manual step progress (see modes._step_callback)
):
    params = dict(
        mode="edit",
        images=[image],
        prompt=prompt,
        speed=speed,
        steps=int(steps),
        true_cfg=float(true_cfg),
        negative_prompt=negative_prompt or " ",
        seed=int(seed),
    )
    return backend.generate_with_retry(_get_backend(), "edit", params, progress)


def on_compose_generate(
    target,
    ref1,
    ref2,
    prompt,
    speed,
    steps,
    true_cfg,
    negative_prompt,
    seed,
    progress=gr.Progress(),  # noqa: B008 — manual step progress (see modes._step_callback)
):
    images = [i for i in (target, ref1, ref2) if i is not None]
    params = dict(
        mode="compose",
        images=images,
        prompt=prompt,
        speed=speed,
        steps=int(steps),
        true_cfg=float(true_cfg),
        negative_prompt=negative_prompt or " ",
        seed=int(seed),
    )
    return backend.generate_with_retry(_get_backend(), "compose", params, progress)


# ----- HTML blocks -----------------------------------------------------------

HEADER_HTML = """
<div style="display:flex;justify-content:space-between;align-items:baseline;padding:8px 0 4px 0;">
  <div style="font-size:16px;font-weight:600;letter-spacing:-0.01em;">
    Qwen Image Editor<span class="qie-brand-period">.</span>
  </div>
  <div class="qie-status-dot" style="font-size:11px;color:#988B7C;letter-spacing:0.02em;">ready</div>
</div>
""".strip()

CTA_HTML = """
<div class="qie-cta">
  Built with care.
  <strong>Star <span class="qie-cta-heart">&#9829;</span> the project</strong> to support it
  <span class="qie-cta-sep">&middot;</span>
  <a href="https://github.com/techfreakworm/qwen-image-editor" target="_blank" rel="noopener noreferrer">GitHub</a>
  <span class="qie-cta-sep">&middot;</span>
  <a href="https://huggingface.co/spaces/techfreakworm/qwen-image-editor" target="_blank" rel="noopener noreferrer">HF Space</a>
</div>
""".strip()


# ----- Blocks ----------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=theme.build_theme(), css=theme.CSS, title="Qwen Image Editor") as demo:
        gr.HTML(HEADER_HTML)
        gr.HTML(CTA_HTML)

        with gr.Tabs():
            with gr.Tab("Edit"):
                e = ui.build_edit_tab()
                e["generate_btn"].click(
                    fn=on_edit_generate,
                    inputs=[
                        e["image"],
                        e["prompt"],
                        e["speed"],
                        e["steps"],
                        e["true_cfg"],
                        e["negative_prompt"],
                        e["seed"],
                    ],
                    outputs=[e["output_image"], e["output_meta"]],
                )
                e["speed"].change(
                    fn=_speed_defaults,
                    inputs=[e["speed"]],
                    outputs=[e["steps"], e["true_cfg"], e["quality_grp"]],
                )

            with gr.Tab("Compose"):
                c = ui.build_compose_tab()
                c["generate_btn"].click(
                    fn=on_compose_generate,
                    inputs=[
                        c["target_image"],
                        c["ref_image_1"],
                        c["ref_image_2"],
                        c["prompt"],
                        c["speed"],
                        c["steps"],
                        c["true_cfg"],
                        c["negative_prompt"],
                        c["seed"],
                    ],
                    outputs=[c["output_image"], c["output_meta"]],
                )
                c["speed"].change(
                    fn=_speed_defaults,
                    inputs=[c["speed"]],
                    outputs=[c["steps"], c["true_cfg"], c["quality_grp"]],
                )

    return demo


# ----- HF ZeroGPU eager startup ----------------------------------------------
# On Spaces, build the pipeline at import (startup) so the `spaces` runtime
# registers the QwenImageEditPlusPipeline's CUDA allocations during the
# supported startup phase — mirroring the official Qwen-Image-Edit-2511 diffusers
# Space. Locally / in CI (not on Spaces) we stay lazy so `import app` needs no torch.
if models.on_spaces():
    _get_backend()


if __name__ == "__main__":
    # default_concurrency_limit=1 → one ZeroGPU task at a time (a 58 GB model can't
    # share a slot; uncapped queueing also spawns multiple GPU workers).
    #
    # ssr_mode=False: disable Gradio 5's experimental server-side rendering. SSR's
    # queue/stream path was masking prediction-function exceptions (the client saw a
    # generic "Error" with NO server traceback + a JS-chunk console error), so errors
    # never surfaced. Plain CSR restores normal error propagation + logging.
    # show_error=True: surface the real exception message in the UI toast (and full
    # traceback in the server logs) instead of a generic "Error".
    build_app().queue(default_concurrency_limit=1).launch(show_error=True, ssr_mode=False)
