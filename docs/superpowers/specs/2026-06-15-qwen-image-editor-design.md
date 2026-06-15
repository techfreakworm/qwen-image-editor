# Qwen Image Editor — Design Spec

> Date: 2026-06-15 · Status: approved (design forks ratified by qwen-brain decision-proxy)

## Goal

A focused, open-source Gradio app that wraps **`Qwen/Qwen-Image-Edit-2511`** for instruction-based image editing and multi-reference composition. Runs locally on NVIDIA CUDA (CPU/MPS degraded) and deploys to a **Hugging Face ZeroGPU** Space. Architecture and conventions **mirror `techfreakworm/z-image-studio`**.

## Non-goals (YAGNI)

- No inpainting / masking, outpainting, ControlNet, or upscaling — not present in the reference workflows.
- No user-supplied LoRA upload — only the built-in Lightning speed LoRA.
- No text-to-image product surface — the model is an editor; an input image is required.
- No multi-model registry / model swapping — single model.

## Capabilities — two tabs, one pipeline

1. **Edit** — one input image + a natural-language instruction → edited image.
2. **Compose** — a target image + up to 2 reference images (material / style / subject) + an instruction → composed edit (the QwenImageEditPlus multi-image conditioning, up to 3 images total).

Both tabs share a single `QwenImageEditPlusPipeline` and a Fast/Quality speed toggle.

## Speed modes

| Mode | LoRA | Scheduler | steps | true_cfg_scale | neg prompt |
|---|---|---|---|---|---|
| **Fast** (default) | Lightning adapter ENABLED | Lightning (shift=ln3, exponential) | 4 | 1.0 | `" "` |
| **Quality** | adapter DISABLED | default (bundled) | 40 | 4.0 | `" "` |

Toggle is implemented via **PEFT adapter enable/disable + scheduler object swap** per request — **no `fuse_lora`/`unfuse_lora` churn** (both are cheap, reversible, ZeroGPU-safe).

## Model & pipeline

- Model: `Qwen/Qwen-Image-Edit-2511` (diffusers-native, ~57.7 GB; bundles transformer / VAE / text_encoder (Qwen2.5-VL-7B) / tokenizer / scheduler).
- Pipeline: `diffusers.QwenImageEditPlusPipeline`, `torch.bfloat16`.
- LoRA: `lightx2v/Qwen-Image-Edit-2511-Lightning`, file `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`, loaded as adapter `"lightning"` (not fused).
- Call: `pipe(image=[imgs], prompt, negative_prompt=" ", true_cfg_scale, num_inference_steps, height, width, generator, output_type="pil").images[0]`.
- Inputs resized to fit ~1 MP, dims rounded to a multiple of 16 (latent patch alignment).

## Architecture — flat module layout (mirror z-image-studio)

| Module | Responsibility |
|---|---|
| `app.py` | Env setup, lazy backend singleton, `gr.Blocks(theme, css)`, two tabs, header/CTA HTML, all event wiring, `queue().launch()` |
| `backend.py` | `QwenImageEditBackend`; optional `import spaces`; `_ON_SPACES`; `_GPU` decorator (dynamic duration); pipeline built once in `__init__` via `_build_pipeline()`; `@_GPU generate(mode, params)`; `generate_with_retry` |
| `modes.py` | Pure handlers `call_edit` / `call_compose` over `(pipe, params)`; `_apply_speed` (scheduler + adapter); generator; returns `(PIL.Image, meta dict)` — no Gradio, no I/O |
| `models.py` | `on_spaces()`, `auto_device()`, model/LoRA id constants, `LIGHTNING_SCHEDULER_CONFIG`, `fit_dimensions()`, local-CUDA offload decision |
| `ui.py` | `build_edit_tab()` / `build_compose_tab()` → `dict[str, Component]`; progressive disclosure; no event wiring |
| `theme.py` | `PALETTE` (violet accent), `build_theme()`, `CSS` (`qie-` prefix) |
| `tooltips.py` | `TOOLTIPS: dict[str, str]` — single source for all `info=` strings |

ZeroGPU wiring mirrors z-image-studio exactly (`try: import spaces`; `_GPU = spaces.GPU(duration=lambda) if on_spaces else identity`; decorator applied at class-definition time; pipeline built once in the backend singleton). **No writable-cache mirror needed** — pure diffusers reads the read-only HF cache fine; `preload_from_hub` handles ZeroGPU preloading.

## Device strategy

- Priority `cuda > mps > cpu`.
- On ZeroGPU (`SPACES_ZERO_GPU` set): build pipeline in the singleton, `.to("cuda")`, **no** cpu offload.
- Local CUDA with < 40 GB free VRAM: `enable_model_cpu_offload()`.
- MPS/CPU: degraded; local dev / UI smoke only (the build box has no GPU).

## UI / Theme

- **"Soft Dark Restraint"**: `#1A1614` substrate, `#F0E8DD` ink, ONE accent = **Qwen violet `#7C5CFF`** (status dot, slider fill, primary button, brand period). `gr.themes.Base` + `.set()` + ~90-line CSS (`qie-` prefix). Inter font. No shadows/gradients (except a faint violet CTA wash). The generated image is the visual focus.
- Each tab: left column = controls, right column = output image + meta JSON.
- Top of each tab: **Fast/Quality** radio that swaps step/cfg defaults and reveals quality-only fields. Header with live-status dot; CTA bar linking GitHub + Space.

## Testing (no local GPU)

- **L1 structural** — files exist, importable, configs/LICENSE correct.
- **L2 mocked** — mode handlers, backend routing, `duration_for`, with a `MagicMock` pipeline (monkeypatch `_build_pipeline`).
- **L3 GPU smoke** — `@pytest.mark.gpu`, skipped by default.
- **L4 Space** — manual UI verification of every path on the deployed ZeroGPU Space.
- `pyproject` `addopts = "-m 'not gpu'"`; CI installs only lightweight deps (no torch/diffusers); **all heavy imports deferred** inside functions so `import <module>` works GPU-free.

## Repo layout / deliverables

`app.py`, `backend.py`, `modes.py`, `models.py`, `ui.py`, `theme.py`, `tooltips.py`, `requirements.txt`, `pyproject.toml`, `README.md` (HF Space frontmatter + docs), `setup.sh`, `LICENSE` (MIT), `.gitignore`, `.github/workflows/ci.yml`, `tests/` (`conftest`, `test_scaffold`, `test_theme`, `test_tooltips`, `test_ui`, `test_models`, `test_modes`, `test_backend`, `test_app`, `test_smoke_gpu`), `docs/superpowers/{specs,plans}`.

## Deploy

- GitHub repo `techfreakworm/qwen-image-editor` (canonical source).
- HF Space `techfreakworm/qwen-image-editor` (gradio sdk, ZeroGPU hardware, `preload_from_hub` for model + LoRA).
- Source mirrored to both remotes.

## Dependencies

- `gradio==5.50.0`; `diffusers @ git+https://github.com/huggingface/diffusers.git@973a077c6a4e7e7a7ea61a84bedd29ac24fb609a` (official Space commit); `transformers`, `accelerate`, `safetensors`, `sentencepiece`, `peft`, `torch>=2.5`, `torchvision`, `pillow`, `numpy` latest-compatible. **Never pin `spaces`** (HF injects it on ZeroGPU). Dev: `ruff`, `pytest`, `pytest-mock`.

## Risks & mitigations

- **57.7 GB cold build** on the Space (slow first build) → `preload_from_hub` + accept one-time build latency.
- **ZeroGPU duration cap** → dynamic `duration_for` clamped `[60,180]`; Fast path well within.
- **No local inference verification** → mocked unit tests locally + real-inference verification on the Space (L4) + a GPU smoke test that runs on the Space.
- **Adapter/scheduler swap correctness** → covered by the L4 Space test across Fast/Quality on both tabs.
