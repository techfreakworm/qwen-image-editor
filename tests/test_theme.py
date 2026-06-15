import gradio as gr

import theme


def test_palette_accent_is_qwen_violet():
    assert theme.PALETTE["accent"] == "#7C5CFF"


def test_palette_accent_text_is_correct():
    assert theme.PALETTE["accent_text"] == "#0E0A1A"


def test_palette_dark_tokens_unchanged():
    pal = theme.PALETTE
    assert pal["body_bg"] == "#1A1614"
    assert pal["text"] == "#F0E8DD"
    assert pal["text_dim"] == "#988B7C"
    assert pal["border"] == "#2A241E"
    assert pal["radius"] == "6px"


def test_build_theme_returns_gradio_base():
    th = theme.build_theme()
    assert isinstance(th, gr.themes.Base)


def test_css_uses_qie_prefix():
    assert ".qie-" in theme.CSS


def test_css_has_no_zis_prefix():
    assert ".zis-" not in theme.CSS


def test_css_has_no_amber_accent():
    assert "#FFB02E" not in theme.CSS


def test_css_includes_soon_row_styling():
    assert ".qie-soon-row" in theme.CSS
    assert ".qie-soon-row a" in theme.CSS


def test_css_includes_compact_file_widget():
    assert ".qie-lora-file" in theme.CSS
    assert "min-height: 56px" in theme.CSS


def test_css_brand_period_uses_violet():
    assert ".qie-brand-period" in theme.CSS
    assert "#7C5CFF" in theme.CSS
