"""Tests for ui.py — component dict structure, progressive disclosure, defaults.

All tests require an active ``gr.Blocks()`` context because Gradio 5 component
constructors register themselves on the current block. The ``_blocks_ctx``
autouse fixture provides that context for every test in this module.
"""

from __future__ import annotations

import gradio as gr
import pytest

import ui


@pytest.fixture(autouse=True)
def _blocks_ctx():
    """Provide a gr.Blocks() context for every test in this module."""
    with gr.Blocks():
        yield


# ---------------------------------------------------------------------------
# Common-key coverage
# ---------------------------------------------------------------------------

_COMMON_KEYS = {
    "prompt",
    "speed",
    "steps",
    "true_cfg",
    "negative_prompt",
    "seed",
    "generate_btn",
    "output_image",
    "output_meta",
    "advanced_grp",
    "quality_grp",
}


def test_edit_tab_returns_all_common_keys():
    result = ui.build_edit_tab()
    assert _COMMON_KEYS.issubset(result.keys())


def test_compose_tab_returns_all_common_keys():
    result = ui.build_compose_tab()
    assert _COMMON_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# Tab-specific keys
# ---------------------------------------------------------------------------


def test_edit_tab_has_image_key():
    result = ui.build_edit_tab()
    assert "image" in result


def test_edit_tab_has_no_compose_image_keys():
    result = ui.build_edit_tab()
    assert "target_image" not in result
    assert "ref_image_1" not in result
    assert "ref_image_2" not in result


def test_compose_tab_has_all_image_keys():
    result = ui.build_compose_tab()
    assert "target_image" in result
    assert "ref_image_1" in result
    assert "ref_image_2" in result


def test_compose_tab_has_no_edit_image_key():
    result = ui.build_compose_tab()
    assert "image" not in result


# ---------------------------------------------------------------------------
# Speed radio
# ---------------------------------------------------------------------------


def _speed_labels(speed_component: gr.Radio) -> list[str]:
    """Extract string labels from a Radio component (Gradio 5 stores tuples)."""
    return [c[0] if isinstance(c, tuple) else c for c in speed_component.choices]


def test_edit_speed_choices():
    result = ui.build_edit_tab()
    assert _speed_labels(result["speed"]) == ["Fast", "Quality"]


def test_edit_speed_default_value():
    result = ui.build_edit_tab()
    assert result["speed"].value == "Fast"


def test_compose_speed_choices():
    result = ui.build_compose_tab()
    assert _speed_labels(result["speed"]) == ["Fast", "Quality"]


def test_compose_speed_default_value():
    result = ui.build_compose_tab()
    assert result["speed"].value == "Fast"


# ---------------------------------------------------------------------------
# Progressive disclosure — quality_grp starts hidden
# ---------------------------------------------------------------------------


def test_edit_quality_grp_starts_hidden():
    result = ui.build_edit_tab()
    assert result["quality_grp"].visible is False


def test_compose_quality_grp_starts_hidden():
    result = ui.build_compose_tab()
    assert result["quality_grp"].visible is False


# ---------------------------------------------------------------------------
# Progressive disclosure — advanced_grp is an Accordion (closed)
# ---------------------------------------------------------------------------


def test_edit_advanced_grp_is_accordion():
    result = ui.build_edit_tab()
    assert isinstance(result["advanced_grp"], gr.Accordion)


def test_compose_advanced_grp_is_accordion():
    result = ui.build_compose_tab()
    assert isinstance(result["advanced_grp"], gr.Accordion)


# ---------------------------------------------------------------------------
# Component types for key outputs
# ---------------------------------------------------------------------------


def test_edit_output_image_type():
    result = ui.build_edit_tab()
    output = result["output_image"]
    assert isinstance(output, gr.Image)
    assert output.type == "pil"


def test_compose_output_image_type():
    result = ui.build_compose_tab()
    output = result["output_image"]
    assert isinstance(output, gr.Image)
    assert output.type == "pil"


def test_edit_output_meta_is_json():
    result = ui.build_edit_tab()
    assert isinstance(result["output_meta"], gr.JSON)


def test_compose_output_meta_is_json():
    result = ui.build_compose_tab()
    assert isinstance(result["output_meta"], gr.JSON)


def test_edit_generate_btn_is_primary():
    result = ui.build_edit_tab()
    assert result["generate_btn"].variant == "primary"


def test_compose_generate_btn_is_primary():
    result = ui.build_compose_tab()
    assert result["generate_btn"].variant == "primary"


# ---------------------------------------------------------------------------
# Input image component types
# ---------------------------------------------------------------------------


def test_edit_image_is_pil():
    result = ui.build_edit_tab()
    assert isinstance(result["image"], gr.Image)
    assert result["image"].type == "pil"


def test_compose_target_image_is_pil():
    result = ui.build_compose_tab()
    assert isinstance(result["target_image"], gr.Image)
    assert result["target_image"].type == "pil"


def test_compose_ref_images_are_pil():
    result = ui.build_compose_tab()
    assert isinstance(result["ref_image_1"], gr.Image)
    assert result["ref_image_1"].type == "pil"
    assert isinstance(result["ref_image_2"], gr.Image)
    assert result["ref_image_2"].type == "pil"
