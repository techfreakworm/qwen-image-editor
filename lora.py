"""Custom user-LoRA resolution + validation.

Security-critical and deliberately runs on the CPU in the request handler (OFF the
@spaces.GPU clock): it resolves a user-supplied LoRA — by HF repo id or by uploaded file —
to a local ``.safetensors`` path, with hard guards suitable for a PUBLIC, credit-backed Space:

* ``.safetensors`` ONLY — pickle formats (.bin/.pt/.ckpt) are refused (arbitrary-code-exec).
* Size cap, checked from repo METADATA *before* downloading (anti-DoS: a repo id pointing at
  a multi-GB model can't be pulled in full).
* On Spaces, user-supplied repos are accessed PUBLIC-only (no owner token) so a user can't
  read the owner's private/gated repos; locally (MPS/CUDA dev) the ambient token is fine.

All failures raise :class:`LoraError` with a friendly, user-facing message — never a crash.
Heavy imports (huggingface_hub) are deferred so importing this module in CI is torch-free.
"""

from __future__ import annotations

import os

GB = 1024**3
MAX_LORA_BYTES = 500 * 1024**2  # LoRAs are ~50-400 MB; cap blocks "point at a full model".
_SAFE_EXT = ".safetensors"


class LoraError(ValueError):
    """User-facing LoRA validation/resolution error (rendered as a friendly UI message)."""


def _is_safetensors(name: str) -> bool:
    return name.lower().endswith(_SAFE_EXT)


def _on_spaces() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def resolve_lora(repo_id: str | None, upload_path: str | None) -> str | None:
    """Resolve a user LoRA to a local ``.safetensors`` path, or ``None`` if none requested.

    Upload takes precedence over a repo id. Runs on CPU (request handler), so the network
    download never burns the billed 2x GPU clock. Raises :class:`LoraError` on any problem.
    """
    repo_id = (repo_id or "").strip()

    if upload_path:
        if not _is_safetensors(upload_path):
            raise LoraError("LoRA upload must be a .safetensors file (pickle formats are refused).")
        try:
            size = os.path.getsize(upload_path)
        except OSError as e:
            raise LoraError(f"Could not read the uploaded LoRA file: {e}") from e
        if size > MAX_LORA_BYTES:
            raise LoraError(
                f"LoRA file is too large ({size / GB:.2f} GB); the limit is {MAX_LORA_BYTES / GB:.1f} GB."
            )
        return upload_path

    if repo_id:
        return _resolve_repo(repo_id)

    return None


def _resolve_repo(repo_id: str) -> str:
    """Find + download the .safetensors LoRA file from an HF repo (size-checked first)."""
    from huggingface_hub import HfApi, hf_hub_download

    # On the public Space: NEVER use the owner's token for a user-supplied repo (would expose
    # the owner's private repos). Locally, the ambient token (the dev's own) is fine.
    token = None if _on_spaces() else (os.environ.get("HF_TOKEN") or None)
    api = HfApi()

    try:
        info = api.model_info(repo_id, files_metadata=True, token=token)
    except Exception as e:
        raise LoraError(
            f"Could not access LoRA repo '{repo_id}' (it must be a public Hugging Face repo): {e}"
        ) from e

    safetensors = [(s.rfilename, s.size or 0) for s in (info.siblings or []) if _is_safetensors(s.rfilename)]
    if not safetensors:
        raise LoraError(f"Repo '{repo_id}' has no .safetensors LoRA file (pickle formats are refused).")

    # Heuristic: the LoRA weights are the largest .safetensors in the repo.
    fname, fsize = max(safetensors, key=lambda t: t[1])
    if fsize > MAX_LORA_BYTES:
        raise LoraError(
            f"LoRA '{fname}' is {fsize / GB:.2f} GB (over the {MAX_LORA_BYTES / GB:.1f} GB limit)."
        )

    try:
        return hf_hub_download(repo_id, fname, token=token)
    except Exception as e:
        raise LoraError(f"Failed to download LoRA '{fname}' from '{repo_id}': {e}") from e
