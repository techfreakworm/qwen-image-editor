"""Tests for app.py — speed defaults and lazy backend instantiation."""

from __future__ import annotations

import gradio as gr

import app


def test_speed_defaults_fast_steps_and_cfg():
    result = app._speed_defaults("Fast")
    assert result[:2] == (4, 1.0)


def test_speed_defaults_quality_steps_and_cfg():
    result = app._speed_defaults("Quality")
    assert result[:2] == (28, 4.0)


def test_speed_defaults_fast_quality_grp_hidden():
    _, _, update = app._speed_defaults("Fast")
    # gr.update() in Gradio 5 returns a dict-like; access via .get or attribute.
    visible = getattr(update, "get", lambda *_: None)("visible")
    if visible is None:
        visible = getattr(update, "visible", None)
    assert visible is False


def test_speed_defaults_quality_quality_grp_visible():
    _, _, update = app._speed_defaults("Quality")
    visible = getattr(update, "get", lambda *_: None)("visible")
    if visible is None:
        visible = getattr(update, "visible", None)
    assert visible is True


def test_build_app_does_not_instantiate_backend(monkeypatch):
    """build_app() must return a gr.Blocks without ever calling _get_backend.

    Proves the deferred-import / lazy-singleton discipline: the heavy pipeline
    construction must not be triggered at UI-build time (only at first generate).
    """

    def _raises():
        raise AssertionError("_get_backend must not be called during build_app()")

    monkeypatch.setattr(app, "_get_backend", _raises)
    demo = app.build_app()
    assert isinstance(demo, gr.Blocks)
