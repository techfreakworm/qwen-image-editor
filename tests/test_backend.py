"""Tests for backend.py — duration estimator, generate routing, and retry logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import backend
import modes

# ---------------------------------------------------------------------------
# Fixture: a QwenImageEditBackend whose _build_pipeline is replaced with a
# MagicMock so no torch/diffusers import occurs.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(backend, "_build_pipeline", lambda *a, **k: MagicMock())
    return backend.QwenImageEditBackend()


# ---------------------------------------------------------------------------
# duration_for — clamping and Fast shortcut
# ---------------------------------------------------------------------------


def test_duration_for_fast_returns_60():
    assert backend.duration_for("edit", {"speed": "Fast"}) == 60


def test_duration_for_fast_ignores_steps():
    """Fast always returns 60 regardless of steps."""
    assert backend.duration_for("compose", {"speed": "Fast", "steps": 100}) == 60


def test_duration_for_quality_default_steps():
    # steps defaults to 40: 30 + 40 * 3.5 = 170, in range [60, 180]
    result = backend.duration_for("edit", {"speed": "Quality"})
    assert result == 170
    assert 60 <= result <= 180


def test_duration_for_quality_40_steps_explicit():
    result = backend.duration_for("edit", {"speed": "Quality", "steps": 40})
    assert result == 170
    assert 60 <= result <= 180


def test_duration_for_clamps_to_minimum():
    # steps=0: 30 + 0 * 3.5 = 30, clamped up to 60
    result = backend.duration_for("edit", {"speed": "Quality", "steps": 0})
    assert result == 60


def test_duration_for_clamps_to_maximum():
    # steps=1000: 30 + 1000 * 3.5 = 3530, clamped down to 180
    result = backend.duration_for("edit", {"speed": "Quality", "steps": 1000})
    assert result == 180


def test_duration_for_quality_gt_fast():
    quality = backend.duration_for("edit", {"speed": "Quality", "steps": 40})
    fast = backend.duration_for("edit", {"speed": "Fast"})
    assert quality > fast


def test_duration_for_returns_int():
    assert isinstance(backend.duration_for("edit", {"speed": "Quality", "steps": 40}), int)
    assert isinstance(backend.duration_for("edit", {"speed": "Fast"}), int)


# ---------------------------------------------------------------------------
# generate — dispatch routing and unknown-mode guard
# ---------------------------------------------------------------------------


def test_generate_routes_to_dispatch(monkeypatch, fake_backend):
    """generate must call modes.DISPATCH[mode] with (pipeline, params)."""
    sentinel = ("output_image", {"meta": True})
    fake_handler = MagicMock(return_value=sentinel)
    monkeypatch.setattr(modes, "DISPATCH", {"edit": fake_handler})

    params = {"speed": "Fast", "prompt": "test"}
    result = fake_backend.generate("edit", params)

    fake_handler.assert_called_once_with(fake_backend.pipeline, params)
    assert result is sentinel


def test_generate_unknown_mode_raises_value_error(monkeypatch, fake_backend):
    """generate must raise ValueError — not KeyError — for unrecognised modes."""
    monkeypatch.setattr(modes, "DISPATCH", {"edit": MagicMock()})
    with pytest.raises(ValueError, match="unknown mode"):
        fake_backend.generate("no_such_mode", {})


# ---------------------------------------------------------------------------
# generate_with_retry — retry on GPU abort, re-raise on other errors
# ---------------------------------------------------------------------------


def test_generate_with_retry_retries_exactly_once_on_gpu_abort(monkeypatch, fake_backend):
    """generate_with_retry retries once when generate raises 'GPU task aborted'."""
    good_result = ("img", {"seed": 0})
    mock_generate = MagicMock(side_effect=[Exception("GPU task aborted"), good_result])
    monkeypatch.setattr(fake_backend, "generate", mock_generate)

    result = backend.generate_with_retry(fake_backend, "edit", {"speed": "Fast"})

    assert mock_generate.call_count == 2
    assert result is good_result


def test_generate_with_retry_case_insensitive_match(monkeypatch, fake_backend):
    """GPU-abort detection is case-insensitive."""
    good_result = ("img", {})
    mock_generate = MagicMock(side_effect=[Exception("GPU Task Aborted"), good_result])
    monkeypatch.setattr(fake_backend, "generate", mock_generate)

    backend.generate_with_retry(fake_backend, "edit", {})
    assert mock_generate.call_count == 2


def test_generate_with_retry_reraises_non_gpu_error(monkeypatch, fake_backend):
    """generate_with_retry must NOT retry on unrelated exceptions."""
    mock_generate = MagicMock(side_effect=Exception("boom"))
    monkeypatch.setattr(fake_backend, "generate", mock_generate)

    with pytest.raises(Exception, match="boom"):
        backend.generate_with_retry(fake_backend, "edit", {})

    assert mock_generate.call_count == 1  # no retry
