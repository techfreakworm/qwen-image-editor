"""Tooltip strings for every labeled UI component.

Kept separate from ``ui.py`` so copy edits don't touch component wiring. Every
key here MUST be referenced from a labeled component in ``ui.py`` (and vice versa).
"""

from __future__ import annotations

TOOLTIPS: dict[str, str] = {
    "prompt": "Natural-language instruction for how to edit the image. Be specific about what should change and what should stay.",
    "image": "The image to edit. Accepts PNG, JPG, or WebP. Large images are auto-resized to fit 1 MP.",
    "target_image": "The base image you want to modify. In Compose mode this is the primary subject being altered.",
    "ref_image": "Reference image supplying material, style, or subject for the composition. Up to 2 references supported.",
    "speed": "Fast uses the Lightning LoRA (4 steps, cfg=1). Quality disables LoRA for 40 steps, higher fidelity.",
    "steps": "Denoising steps. Fast: 4 (Lightning LoRA). Quality: 40. More steps increase detail and run time.",
    "true_cfg": "Classifier-free guidance scale. Fast: 1.0. Quality: 4.0. Higher values follow the prompt more strictly.",
    "negative_prompt": "Concepts to steer away from. Leave blank to use the model default. More effective in Quality mode.",
    "seed": "-1 picks a random seed each run. Set a fixed integer to reproduce an image exactly.",
}
