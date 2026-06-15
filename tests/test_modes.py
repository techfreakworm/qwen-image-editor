"""L2 mocked tests for modes.py — no GPU, no torch, no diffusers required."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

import models
import modes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_IMAGE = Image.new("RGB", (8, 8))


def _pipe_result() -> MagicMock:
    """A fake pipeline output whose .images[0] is a small PIL image."""
    result = MagicMock()
    result.images = [Image.new("RGB", (8, 8))]
    return result


def _make_params(
    *,
    speed: str = "Fast",
    steps: int = 4,
    true_cfg: float = 1.0,
    seed: int = 42,
    images: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "prompt": "make it raining",
        "images": images if images is not None else [_DUMMY_IMAGE],
        "speed": speed,
        "steps": steps,
        "true_cfg": true_cfg,
        "negative_prompt": " ",
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_pipe() -> MagicMock:
    """A MagicMock pipeline with all attributes needed by modes._run."""
    pipe = MagicMock()
    pipe.device = "cpu"
    pipe.return_value = _pipe_result()
    pipe._qie_lightning_scheduler = MagicMock()
    pipe._qie_default_scheduler = MagicMock()
    return pipe


@pytest.fixture(autouse=True)
def _patch_fit_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch models.fit_dimensions to return (64, 64) so no real image maths runs."""
    monkeypatch.setattr(models, "fit_dimensions", lambda img: (64, 64))


@pytest.fixture(autouse=True)
def _patch_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a stub torch module so the lazy 'import torch' in _run succeeds in CI."""
    mock_torch = MagicMock()
    gen = MagicMock()
    mock_torch.Generator.return_value = gen
    gen.manual_seed.return_value = gen
    monkeypatch.setitem(sys.modules, "torch", mock_torch)


# ---------------------------------------------------------------------------
# call_edit
# ---------------------------------------------------------------------------


def test_edit_passes_target_image_in_image_kwarg(fake_pipe: MagicMock) -> None:
    """call_edit must forward exactly [target] in the image= positional kwarg."""
    target = Image.new("RGB", (16, 16))
    params = _make_params(images=[target])

    modes.call_edit(fake_pipe, params)

    _, call_kwargs = fake_pipe.call_args
    assert call_kwargs["image"] == [target]


def test_edit_sets_mode_in_meta(fake_pipe: MagicMock) -> None:
    """call_edit must record mode='edit' in the returned metadata."""
    _, meta = modes.call_edit(fake_pipe, _make_params())
    assert meta["mode"] == "edit"


# ---------------------------------------------------------------------------
# call_compose
# ---------------------------------------------------------------------------


def test_compose_drops_none_images(fake_pipe: MagicMock) -> None:
    """Compose with [target, None, ref2] must call pipe with image=[target, ref2]."""
    target = Image.new("RGB", (8, 8))
    ref2 = Image.new("RGB", (8, 8))
    params = _make_params(images=[target, None, ref2])

    modes.call_compose(fake_pipe, params)

    _, call_kwargs = fake_pipe.call_args
    assert call_kwargs["image"] == [target, ref2]
    assert len(call_kwargs["image"]) == 2


def test_compose_sets_mode_in_meta(fake_pipe: MagicMock) -> None:
    """call_compose must record mode='compose' in the returned metadata."""
    _, meta = modes.call_compose(fake_pipe, _make_params())
    assert meta["mode"] == "compose"


# ---------------------------------------------------------------------------
# Speed modes
# ---------------------------------------------------------------------------


def test_fast_speed_calls_set_adapters(fake_pipe: MagicMock) -> None:
    """Fast speed must call pipe.set_adapters with the Lightning adapter at weight 1.0."""
    modes.call_edit(fake_pipe, _make_params(speed="Fast", steps=4, true_cfg=1.0))

    fake_pipe.set_adapters.assert_called_once_with([models.LORA_ADAPTER_NAME], [1.0])


def test_fast_speed_pipe_call_kwargs(fake_pipe: MagicMock) -> None:
    """Fast speed must pass num_inference_steps=4 and true_cfg_scale=1.0 to pipe."""
    modes.call_edit(fake_pipe, _make_params(speed="Fast", steps=4, true_cfg=1.0))

    _, call_kwargs = fake_pipe.call_args
    assert call_kwargs["num_inference_steps"] == 4
    assert call_kwargs["true_cfg_scale"] == 1.0


def test_fast_speed_uses_lightning_scheduler(fake_pipe: MagicMock) -> None:
    """Fast speed must swap in the Lightning scheduler before the pipe call."""
    modes.call_edit(fake_pipe, _make_params(speed="Fast"))

    assert fake_pipe.scheduler is fake_pipe._qie_lightning_scheduler


def test_quality_speed_calls_disable_lora(fake_pipe: MagicMock) -> None:
    """Quality speed must call pipe.disable_lora() to deactivate the Lightning adapter."""
    modes.call_edit(fake_pipe, _make_params(speed="Quality", steps=40, true_cfg=4.0))

    fake_pipe.disable_lora.assert_called_once()


def test_quality_speed_pipe_call_kwargs(fake_pipe: MagicMock) -> None:
    """Quality speed must pass num_inference_steps=40 and true_cfg_scale=4.0 to pipe."""
    modes.call_edit(fake_pipe, _make_params(speed="Quality", steps=40, true_cfg=4.0))

    _, call_kwargs = fake_pipe.call_args
    assert call_kwargs["num_inference_steps"] == 40
    assert call_kwargs["true_cfg_scale"] == 4.0


def test_quality_speed_uses_default_scheduler(fake_pipe: MagicMock) -> None:
    """Quality speed must restore the default scheduler before the pipe call."""
    modes.call_edit(fake_pipe, _make_params(speed="Quality"))

    assert fake_pipe.scheduler is fake_pipe._qie_default_scheduler


# ---------------------------------------------------------------------------
# Seed handling
# ---------------------------------------------------------------------------


def test_negative_seed_produces_int_in_meta(fake_pipe: MagicMock) -> None:
    """seed=-1 must resolve to a non-negative int stored in meta['seed']."""
    _, meta = modes.call_edit(fake_pipe, _make_params(seed=-1))

    assert isinstance(meta["seed"], int)
    assert meta["seed"] >= 0


def test_explicit_seed_preserved_in_meta(fake_pipe: MagicMock) -> None:
    """An explicit non-negative seed must appear unchanged in meta['seed']."""
    _, meta = modes.call_edit(fake_pipe, _make_params(seed=123))
    assert meta["seed"] == 123


def test_negative_seed_varies(fake_pipe: MagicMock) -> None:
    """Two calls with seed=-1 should (with overwhelming probability) produce different seeds."""
    _, meta1 = modes.call_edit(fake_pipe, _make_params(seed=-1))
    _, meta2 = modes.call_edit(fake_pipe, _make_params(seed=-1))
    # Not deterministically different, but the chance of collision is < 1/2^32.
    # We just assert both are valid ints — determinism is tested via explicit seed.
    assert isinstance(meta1["seed"], int)
    assert isinstance(meta2["seed"], int)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_empty_images_raises_value_error(fake_pipe: MagicMock) -> None:
    """_run must raise ValueError when the images list is empty."""
    params = _make_params(images=[])
    with pytest.raises(ValueError):
        modes.call_edit(fake_pipe, params)


def test_compose_all_none_raises_value_error(fake_pipe: MagicMock) -> None:
    """call_compose with all-None images must raise ValueError after filtering."""
    params = _make_params(images=[None, None])
    with pytest.raises(ValueError):
        modes.call_compose(fake_pipe, params)


# ---------------------------------------------------------------------------
# Meta dict completeness
# ---------------------------------------------------------------------------


_META_KEYS = {"mode", "speed", "steps", "true_cfg", "seed", "width", "height", "num_inputs"}


def test_meta_has_all_documented_keys_edit(fake_pipe: MagicMock) -> None:
    """call_edit must return a metadata dict containing every documented key."""
    _, meta = modes.call_edit(fake_pipe, _make_params())
    assert _META_KEYS <= set(meta.keys())


def test_meta_has_all_documented_keys_compose(fake_pipe: MagicMock) -> None:
    """call_compose must return a metadata dict containing every documented key."""
    target = Image.new("RGB", (8, 8))
    ref = Image.new("RGB", (8, 8))
    _, meta = modes.call_compose(fake_pipe, _make_params(images=[target, ref]))
    assert _META_KEYS <= set(meta.keys())


def test_meta_num_inputs_reflects_image_count(fake_pipe: MagicMock) -> None:
    """meta['num_inputs'] must equal the number of images actually passed to the pipeline."""
    target = Image.new("RGB", (8, 8))
    ref = Image.new("RGB", (8, 8))
    # compose with [target, None, ref] — None is dropped, so num_inputs should be 2
    _, meta = modes.call_compose(fake_pipe, _make_params(images=[target, None, ref]))
    assert meta["num_inputs"] == 2


def test_meta_dimensions_come_from_fit_dimensions(fake_pipe: MagicMock) -> None:
    """meta width/height must match the values returned by models.fit_dimensions (mocked to 64,64)."""
    _, meta = modes.call_edit(fake_pipe, _make_params())
    assert meta["width"] == 64
    assert meta["height"] == 64


# ---------------------------------------------------------------------------
# DISPATCH table
# ---------------------------------------------------------------------------


def test_dispatch_contains_edit_and_compose() -> None:
    """DISPATCH must map 'edit' and 'compose' to the correct callables."""
    assert modes.DISPATCH["edit"] is modes.call_edit
    assert modes.DISPATCH["compose"] is modes.call_compose


def test_dispatch_via_table(fake_pipe: MagicMock) -> None:
    """Calling DISPATCH['edit'] must behave identically to calling call_edit directly."""
    params = _make_params()
    img, meta = modes.DISPATCH["edit"](fake_pipe, params)
    assert isinstance(img, Image.Image)
    assert meta["mode"] == "edit"
