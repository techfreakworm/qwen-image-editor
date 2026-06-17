"""Gradio UI builders for qwen-image-editor (Soft Dark Restraint — violet accent).

Each ``build_*_tab()`` returns a ``dict[str, gr.components.Component]`` so
``app.py:build_app`` can wire ``.click()`` / ``.change()`` handlers without
reaching into local scopes.  No event wiring lives here.

All param help text flows through Gradio's native ``info=`` parameter (dim
subtitle under each label).  ``gr.Image`` does not support ``info=``, so those
components use a descriptive label instead.
"""

from __future__ import annotations

import gradio as gr

from tooltips import TOOLTIPS

# Common slider bounds shared across both tabs.
_STEPS_MIN = 1
_STEPS_MAX = 50
_STEPS_DEFAULT = 4  # Fast mode default; app.py swaps to 40 on Quality.

_CFG_MIN = 0.5
_CFG_MAX = 12.0
_CFG_DEFAULT = 1.0  # Fast mode default; app.py swaps to 4.0 on Quality.

_LORA_PLACEHOLDER = "e.g. fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"


def _lora_controls() -> dict[str, gr.components.Component]:
    """Custom-LoRA controls (repo id + upload + weight), shared by both tabs.

    Custom LoRAs apply in **Quality** mode only — in Fast mode the Lightning 4-step
    distillation fights a content LoRA, so it's intentionally ignored there.
    """
    with gr.Accordion("Custom LoRA (Quality mode)", open=False):
        gr.Markdown(
            "Apply a custom LoRA on top of the base model. **Quality mode only** — ignored in Fast. "
            "Leave blank for the default model.",
        )
        lora_repo = gr.Textbox(
            label="LoRA — Hugging Face repo id",
            placeholder=_LORA_PLACEHOLDER,
            lines=1,
        )
        lora_file = gr.File(
            label="…or upload a LoRA (.safetensors only)",
            file_types=[".safetensors"],
            file_count="single",
        )
        lora_weight = gr.Slider(
            0.0, 1.5, value=0.9, step=0.05,
            label="LoRA weight",
            info="Strength of the custom LoRA (0 = off, 1.0 = full).",
        )
    return dict(lora_repo=lora_repo, lora_file=lora_file, lora_weight=lora_weight)


def build_edit_tab() -> dict[str, gr.components.Component]:
    """Build the Edit tab layout.

    Returns a flat dict of all components.  Keys:
    ``prompt, speed, steps, true_cfg, negative_prompt, seed, generate_btn,
    output_image, output_meta, advanced_grp, quality_grp, image``.
    """
    with gr.Row():
        with gr.Column(scale=4):
            image = gr.Image(
                label="Input image",
                type="pil",
                height=300,
            )

            prompt = gr.Textbox(
                label="Prompt",
                info=TOOLTIPS["prompt"],
                lines=3,
                placeholder="Make the sky dramatic with orange-purple sunset hues…",
            )

            speed = gr.Radio(
                ["Fast", "Quality"],
                value="Fast",
                label="Speed",
                info=TOOLTIPS["speed"],
            )

            with gr.Group(visible=False) as quality_grp:
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    info=TOOLTIPS["negative_prompt"],
                    lines=2,
                    placeholder="blurry, lowres, distorted",
                )
                true_cfg = gr.Slider(
                    _CFG_MIN,
                    _CFG_MAX,
                    value=_CFG_DEFAULT,
                    step=0.1,
                    label="CFG scale",
                    info=TOOLTIPS["true_cfg"],
                )

            with gr.Accordion("Advanced", open=False) as advanced_grp:
                steps = gr.Slider(
                    _STEPS_MIN,
                    _STEPS_MAX,
                    value=_STEPS_DEFAULT,
                    step=1,
                    label="Steps",
                    info=TOOLTIPS["steps"],
                )
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed",
                    info=TOOLTIPS["seed"],
                )

            lora = _lora_controls()

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=5):
            output_image = gr.Image(
                show_label=False,
                type="pil",
                height=512,
                show_download_button=True,
            )
            output_meta = gr.JSON(label="Meta", value={})

    return dict(
        image=image,
        prompt=prompt,
        speed=speed,
        quality_grp=quality_grp,
        negative_prompt=negative_prompt,
        true_cfg=true_cfg,
        advanced_grp=advanced_grp,
        steps=steps,
        seed=seed,
        generate_btn=generate_btn,
        output_image=output_image,
        output_meta=output_meta,
        **lora,
    )


def build_compose_tab() -> dict[str, gr.components.Component]:
    """Build the Compose tab layout.

    Returns a flat dict of all components.  Keys:
    ``prompt, speed, steps, true_cfg, negative_prompt, seed, generate_btn,
    output_image, output_meta, advanced_grp, quality_grp,
    target_image, ref_image_1, ref_image_2``.
    """
    with gr.Row():
        with gr.Column(scale=4):
            target_image = gr.Image(
                label="Target image",
                type="pil",
                height=240,
            )

            with gr.Row():
                ref_image_1 = gr.Image(
                    label="Reference 1",
                    type="pil",
                    height=200,
                )
                ref_image_2 = gr.Image(
                    label="Reference 2 (optional)",
                    type="pil",
                    height=200,
                )

            prompt = gr.Textbox(
                label="Prompt",
                info=TOOLTIPS["prompt"],
                lines=3,
                placeholder="Wearing the jacket from the reference, same background…",
            )

            speed = gr.Radio(
                ["Fast", "Quality"],
                value="Fast",
                label="Speed",
                info=TOOLTIPS["speed"],
            )

            with gr.Group(visible=False) as quality_grp:
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    info=TOOLTIPS["negative_prompt"],
                    lines=2,
                    placeholder="blurry, lowres, distorted",
                )
                true_cfg = gr.Slider(
                    _CFG_MIN,
                    _CFG_MAX,
                    value=_CFG_DEFAULT,
                    step=0.1,
                    label="CFG scale",
                    info=TOOLTIPS["true_cfg"],
                )

            with gr.Accordion("Advanced", open=False) as advanced_grp:
                steps = gr.Slider(
                    _STEPS_MIN,
                    _STEPS_MAX,
                    value=_STEPS_DEFAULT,
                    step=1,
                    label="Steps",
                    info=TOOLTIPS["steps"],
                )
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed",
                    info=TOOLTIPS["seed"],
                )

            lora = _lora_controls()

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=5):
            output_image = gr.Image(
                show_label=False,
                type="pil",
                height=512,
                show_download_button=True,
            )
            output_meta = gr.JSON(label="Meta", value={})

    return dict(
        target_image=target_image,
        ref_image_1=ref_image_1,
        ref_image_2=ref_image_2,
        prompt=prompt,
        speed=speed,
        quality_grp=quality_grp,
        negative_prompt=negative_prompt,
        true_cfg=true_cfg,
        advanced_grp=advanced_grp,
        steps=steps,
        seed=seed,
        generate_btn=generate_btn,
        output_image=output_image,
        output_meta=output_meta,
        **lora,
    )
