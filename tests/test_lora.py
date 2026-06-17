"""Tests for lora.resolve_lora — the security-critical user-LoRA resolver."""

from __future__ import annotations

import types

import pytest

import lora


def test_no_lora_returns_none():
    assert lora.resolve_lora(None, None) is None
    assert lora.resolve_lora("", None) is None
    assert lora.resolve_lora("   ", None) is None


def test_upload_rejects_non_safetensors(tmp_path):
    bad = tmp_path / "evil.bin"
    bad.write_bytes(b"\x00\x01")
    with pytest.raises(lora.LoraError, match="safetensors"):
        lora.resolve_lora(None, str(bad))


def test_upload_accepts_safetensors(tmp_path):
    good = tmp_path / "my_lora.safetensors"
    good.write_bytes(b"\x00" * 1024)
    assert lora.resolve_lora(None, str(good)) == str(good)


def test_upload_rejects_oversized(tmp_path, monkeypatch):
    big = tmp_path / "huge.safetensors"
    big.write_bytes(b"\x00" * 16)
    monkeypatch.setattr(lora.os.path, "getsize", lambda _p: lora.MAX_LORA_BYTES + 1)
    with pytest.raises(lora.LoraError, match="too large"):
        lora.resolve_lora(None, str(big))


def test_upload_precedence_over_repo(tmp_path):
    good = tmp_path / "up.safetensors"
    good.write_bytes(b"\x00" * 8)
    # Upload wins; the repo id is not even consulted (no network).
    assert lora.resolve_lora("some/repo", str(good)) == str(good)


def _fake_model_info(files):
    """files: list of (rfilename, size) -> a stub ModelInfo with .siblings."""
    sibs = [types.SimpleNamespace(rfilename=n, size=s) for n, s in files]
    return types.SimpleNamespace(siblings=sibs)


def test_repo_downloads_safetensors(mocker):
    mocker.patch("huggingface_hub.HfApi.model_info", return_value=_fake_model_info(
        [("README.md", 10), ("lora.safetensors", 200 * 1024**2)]
    ))
    dl = mocker.patch("huggingface_hub.hf_hub_download", return_value="/cache/lora.safetensors")
    out = lora.resolve_lora("fal/Some-LoRA", None)
    assert out == "/cache/lora.safetensors"
    dl.assert_called_once()
    assert dl.call_args.args[1] == "lora.safetensors"  # picked the .safetensors, not README


def test_repo_without_safetensors_refused(mocker):
    mocker.patch("huggingface_hub.HfApi.model_info", return_value=_fake_model_info(
        [("pytorch_lora_weights.bin", 100 * 1024**2)]  # pickle format
    ))
    with pytest.raises(lora.LoraError, match="safetensors"):
        lora.resolve_lora("some/pickle-only", None)


def test_repo_oversized_file_refused_before_download(mocker):
    mocker.patch("huggingface_hub.HfApi.model_info", return_value=_fake_model_info(
        [("full_model.safetensors", 58 * 1024**3)]  # 58 GB "LoRA" -> DoS
    ))
    dl = mocker.patch("huggingface_hub.hf_hub_download")
    with pytest.raises(lora.LoraError, match="limit"):
        lora.resolve_lora("someone/huge", None)
    dl.assert_not_called()  # refused BEFORE downloading


def test_repo_inaccessible_friendly_error(mocker):
    mocker.patch("huggingface_hub.HfApi.model_info", side_effect=OSError("404"))
    with pytest.raises(lora.LoraError, match="public"):
        lora.resolve_lora("nope/missing", None)


def test_spaces_uses_no_token_for_user_repo(mocker, monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    monkeypatch.setenv("HF_TOKEN", "owner-secret")
    mi = mocker.patch("huggingface_hub.HfApi.model_info", return_value=_fake_model_info(
        [("lora.safetensors", 100 * 1024**2)]
    ))
    mocker.patch("huggingface_hub.hf_hub_download", return_value="/cache/lora.safetensors")
    lora.resolve_lora("user/repo", None)
    # On Spaces, the owner's token must NOT be used for a user-supplied repo.
    assert mi.call_args.kwargs.get("token") is None
