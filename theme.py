"""Soft Dark Restraint theme — warm dark surface with a single violet accent.

Palette tokens, ``gr.themes.Base`` configuration, and a small CSS string that
applies the touches Gradio's theme tokens can't express alone (link row under
a model selector + compact file widget).
"""

from __future__ import annotations

import gradio as gr

# Single source of truth for the palette. Variant: "Soft Dark Restraint" with
# Qwen violet accent replacing the amber from z-image-studio.
PALETTE: dict[str, str] = {
    "body_bg": "#1A1614",
    "panel_bg": "#1F1B17",
    "input_bg": "#14110F",
    "text": "#F0E8DD",
    "text_dim": "#988B7C",
    "border": "#2A241E",
    "border_strong": "#3A3128",
    "accent": "#7C5CFF",
    "accent_text": "#0E0A1A",
    "radius": "6px",
}


def build_theme() -> gr.themes.Base:
    """Return a Gradio theme matching the Soft Dark Restraint palette.

    Uses Gradio's default font chain (Inter + system-ui fallbacks). No mono
    font — the redesign drops monospaced UI text entirely.
    """
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#F1EEFF",
            c100="#E0D5FF",
            c200="#C4ADFF",
            c300="#A885FF",
            c400="#8E6EFF",
            c500=PALETTE["accent"],
            c600="#6344E6",
            c700="#4D33C0",
            c800="#38249A",
            c900="#221666",
            c950="#0E0A1A",
        ),
        neutral_hue=gr.themes.Color(
            c50="#F0E8DD",
            c100="#E0D5C3",
            c200="#C8B89E",
            c300="#988B7C",
            c400="#7A6E60",
            c500="#5C5246",
            c600="#3A3128",
            c700="#2A241E",
            c800="#1F1B17",
            c900="#1A1614",
            c950="#14110F",
        ),
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill=PALETTE["body_bg"],
        body_text_color=PALETTE["text"],
        body_text_color_subdued=PALETTE["text_dim"],
        background_fill_primary=PALETTE["panel_bg"],
        background_fill_secondary=PALETTE["body_bg"],
        block_background_fill=PALETTE["panel_bg"],
        block_border_color=PALETTE["border"],
        block_border_width="1px",
        block_radius=PALETTE["radius"],
        input_background_fill=PALETTE["input_bg"],
        input_border_color=PALETTE["border"],
        button_primary_background_fill=PALETTE["accent"],
        button_primary_background_fill_hover=PALETTE["accent"],
        button_primary_text_color=PALETTE["accent_text"],
        button_primary_border_color=PALETTE["accent"],
        slider_color=PALETTE["accent"],
        color_accent=PALETTE["accent"],
        color_accent_soft="rgba(124,92,255,0.12)",
    )


CSS: str = """
/* Soft Dark Restraint — calm, single-accent decorations Gradio tokens can't express. */

/* Small dim link row under the model selector (variant / coming soon links). */
.qie-soon-row {
    margin-top: 6px;
    font-size: 12px;
    color: #988B7C;
    line-height: 1.45;
}
.qie-soon-row a {
    color: #988B7C;
    text-decoration: underline;
    text-decoration-color: #3A3128;
    text-underline-offset: 3px;
}
.qie-soon-row a:hover {
    color: #F0E8DD;
    text-decoration-color: #988B7C;
}
.qie-soon-row .sep { margin: 0 6px; color: #3A3128; }
.qie-soon-row .dim { margin-left: 6px; color: #635A4E; }

/* Compact file widget — tighten Gradio's default 400px drop zone. */
.qie-lora-file .upload-container { min-height: 56px !important; padding: 8px 12px !important; }
.qie-lora-file .icon-wrap, .qie-lora-file svg { display: none !important; }
.qie-lora-file .wrap > * { text-align: left; }

/* Brand period uses the accent. */
.qie-brand-period { color: #7C5CFF; }

/* Live-status dot. */
.qie-status-dot::before {
    content: "";
    display: inline-block;
    width: 6px; height: 6px; border-radius: 50%;
    background: #7C5CFF;
    margin-right: 6px;
    vertical-align: middle;
}

/* CTA bar. */
.qie-cta {
    margin: 4px 0 10px 0;
    padding: 8px 14px;
    font-size: 12px;
    line-height: 1.5;
    color: #988B7C;
    background: linear-gradient(180deg, rgba(124,92,255,0.05), rgba(124,92,255,0.02));
    border: 1px solid #2A241E;
    border-radius: 8px;
    text-align: center;
}
.qie-cta strong { color: #F0E8DD; font-weight: 600; }
.qie-cta .qie-cta-heart {
    display: inline-block;
    color: #7C5CFF;
    transform: translateY(-1px);
    margin: 0 1px;
    animation: qie-cta-pulse 2.4s ease-in-out infinite;
}
@keyframes qie-cta-pulse {
    0%, 60%, 100% { transform: translateY(-1px) scale(1); }
    30% { transform: translateY(-1px) scale(1.18); }
}
.qie-cta .qie-cta-sep { margin: 0 10px; color: #3A3128; }
.qie-cta a {
    color: #7C5CFF;
    text-decoration: none;
    border-bottom: 1px dashed rgba(124,92,255,0.4);
    transition: border-color 0.15s, color 0.15s;
}
.qie-cta a:hover { color: #9B82FF; border-bottom-color: #7C5CFF; }

@media (max-width: 600px) {
    .qie-cta { font-size: 11px; padding: 8px 10px; }
    .qie-cta .qie-cta-sep { display: block; height: 4px; margin: 0; visibility: hidden; }
}
""".strip()
