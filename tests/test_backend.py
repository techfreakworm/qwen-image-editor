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


# The Space runs on the xlarge ZeroGPU tier (58 GB model), which DOUBLES the
# requested duration for ZeroGPU's per-call ceiling check. Requests at/above the
# ceiling are rejected ("ZeroGPU illegal duration"). These tests lock in the two
# invariants that keep the GPU path working, rather than a magic number.
_XLARGE_MULTIPLIER = 2
_ZEROGPU_CEILING = 300  # requested-seconds; reaching this is rejected on this Space

_DURATION_CASES = (
    {"speed": "Fast"},
    {"speed": "Fast", "steps": 100},
    {"speed": "Quality"},
    {"speed": "Quality", "steps": 40},
    {"speed": "Quality", "steps": 1000},
)


def test_duration_for_legal_on_xlarge():
    """The doubled request must stay under ZeroGPU's per-call ceiling.

    This is the regression guard against the illegal-duration bug: a 180 s
    request became 360 s on xlarge and every call was rejected.
    """
    for params in _DURATION_CASES:
        d = backend.duration_for("edit", params)
        assert d * _XLARGE_MULTIPLIER < _ZEROGPU_CEILING, (
            f"duration {d}s -> {d * _XLARGE_MULTIPLIER}s requested >= {_ZEROGPU_CEILING}s ceiling (illegal on xlarge)"
        )


def test_duration_for_covers_materialization():
    """Budget must cover the ~15-40 s per-call model materialization plus
    inference, else the GPU task aborts mid-load."""
    for params in _DURATION_CASES:
        assert backend.duration_for("edit", params) >= 50


def test_duration_for_returns_legal_int():
    """Default budget is a positive int whose doubled xlarge request stays under
    the 360 s value ZeroGPU rejected."""
    d = backend.duration_for("edit", {"speed": "Fast", "steps": 4})
    assert isinstance(d, int)
    assert 60 <= d * 2 < 360


def test_duration_for_env_override(monkeypatch):
    """QIE_GPU_DURATION retunes the budget without a code change; junk is ignored."""
    monkeypatch.setenv("QIE_GPU_DURATION", "100")
    assert backend.duration_for("edit", {"speed": "Fast", "steps": 4}) == 100
    monkeypatch.setenv("QIE_GPU_DURATION", "not-an-int")
    assert isinstance(backend.duration_for("edit", {"speed": "Fast"}), int)


def test_duration_for_returns_int():
    assert isinstance(backend.duration_for("edit", {"speed": "Quality", "steps": 40}), int)
    assert isinstance(backend.duration_for("edit", {"speed": "Fast"}), int)


# ---------------------------------------------------------------------------
# generate — dispatch routing and unknown-mode guard
# ---------------------------------------------------------------------------


def test_generate_routes_to_dispatch(monkeypatch, fake_backend):
    """generate must call modes.DISPATCH[mode] with (pipeline, params, progress)."""
    sentinel = ("output_image", {"meta": True})
    fake_handler = MagicMock(return_value=sentinel)
    monkeypatch.setattr(modes, "DISPATCH", {"edit": fake_handler})

    params = {"speed": "Fast", "prompt": "test"}
    result = fake_backend.generate("edit", params)

    # progress defaults to None when not supplied (e.g. non-UI callers / CI).
    fake_handler.assert_called_once_with(fake_backend.pipeline, params, None)
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
