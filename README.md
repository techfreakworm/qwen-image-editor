---
title: Qwen Image Editor
emoji: 🎨
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
python_version: "3.11"
suggested_hardware: zero-a10g
hf_oauth: false
preload_from_hub:
  - Qwen/Qwen-Image-Edit-2511 transformer/*,vae/*,text_encoder/*,tokenizer/*,scheduler/*,processor/*,model_index.json
  - lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
---

# Qwen Image Editor

A focused Gradio app that wraps [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) under two tabs — **Edit** (single-image instruction following) and **Compose** (blend up to three reference images into one scene). Runs locally on NVIDIA CUDA, deploys to Hugging Face Spaces (ZeroGPU). Fast mode drives the [Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) at 4 steps; Quality mode runs the full 40-step pipeline.

[![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Live-7C5CFF?style=flat-square)](https://huggingface.co/spaces/techfreakworm/qwen-image-editor)
[![GitHub stars](https://img.shields.io/github/stars/techfreakworm/qwen-image-editor?style=flat-square&color=7C5CFF)](https://github.com/techfreakworm/qwen-image-editor/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-7C5CFF?style=flat-square)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-7C5CFF?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Backend: diffusers](https://img.shields.io/badge/backend-diffusers-7C5CFF?style=flat-square)](https://github.com/huggingface/diffusers)

→ **Live demo:** https://huggingface.co/spaces/techfreakworm/qwen-image-editor

---

## What's inside

Two tabs. One `QwenImageEditPlusPipeline` underneath. Progressive disclosure — the form starts short and reveals quality controls only when you need them.

| Tab | What it does |
|---|---|
| **Edit** | Drop one image and describe the change. The model follows natural-language instructions ("add snow", "change the sky to night", "make it a watercolour painting"). |
| **Compose** | Supply a target image plus up to two reference images and describe the scene. The model blends all inputs into a single coherent output. |

Speed is controlled by a **Fast / Quality** toggle present on every tab:

| Mode | Steps | CFG (`true_cfg_scale`) | Notes |
|---|---|---|---|
| **Fast** (Lightning) | 4 | 1.0 | Lightning LoRA active; no CFG double-pass; ~10x faster than quality on H200. |
| **Quality** | 40 | 4.0 | Lightning LoRA disabled; standard `FlowMatchEulerDiscreteScheduler`; full fidelity. |

---

## Quick start (local)

Requires **Python 3.11**, ~58 GB free disk for the model weights, and ~40 GB VRAM peak (CUDA). For GPUs with less VRAM, the backend falls back to `enable_model_cpu_offload` automatically.

```bash
git clone https://github.com/techfreakworm/qwen-image-editor
cd qwen-image-editor
bash setup.sh           # creates .venv, installs requirements
source .venv/bin/activate
python app.py           # http://127.0.0.1:7860
```

Weights are downloaded to `~/.cache/huggingface/hub/` on first run. Subsequent starts skip the download.

## Quick start (HF Spaces)

```bash
git remote add space https://huggingface.co/spaces/techfreakworm/qwen-image-editor
git push space main
```

The Space's `preload_from_hub` directive pre-downloads the ~57.7 GB weight set at build time. The pipeline is ready on first inference — no cold-download penalty at request time.

## Architecture

```
              browser
                 │
                 ▼
   ┌─────────────────────────────┐
   │   app.py — Gradio Blocks    │
   │   (header + CTA + 2 tabs)  │
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │   backend.py                │
   │   QwenImageEditBackend      │
   │   @spaces.GPU(duration=…)   │
   │   single pipeline instance  │
   └──────────────┬──────────────┘
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
   modes.py             models.py
   call_edit /          device autodetect
   call_compose         fit_dimensions
                        offload helpers
```

**Single pipeline instance** (`QwenImageEditPlusPipeline`), constructed once at startup and shared across all requests. Fast / Quality switching is done in-place per request:

- **Fast:** `pipe.set_adapters(["lightning"], [1.0])` enables the preloaded PEFT LoRA adapter, then `pipe.scheduler` is swapped to the Lightning-tuned `FlowMatchEulerDiscreteScheduler` (base_shift=log(3), exponential time shifting). Steps = 4, `true_cfg_scale` = 1.0 (no CFG double-pass needed for distilled model).
- **Quality:** `pipe.disable_lora()` and `pipe.scheduler` is restored to the default scheduler. Steps = 40, `true_cfg_scale` = 4.0.

`@spaces.GPU(duration=callable)` is applied to `QwenImageEditBackend.generate` at module load time. The duration estimator returns **60 s** for Fast and clamps Quality to **[60, 180] s** based on step count. A "GPU task aborted" exception triggers one automatic retry at 2× duration.

## Project layout

```
.
├── app.py              # Gradio Blocks entry; lazy backend singleton; event wiring; CTA
├── backend.py          # QwenImageEditBackend; @spaces.GPU; duration estimator; retry
├── modes.py            # call_edit / call_compose — pure handlers over pipeline + params
├── models.py           # device autodetect; fit_dimensions; offload helpers; constants
├── ui.py               # build_edit_tab / build_compose_tab — component builders
├── theme.py            # Soft Dark Restraint palette (violet accent) + minimal CSS
├── tooltips.py         # Centralised info= strings for every Gradio component
├── requirements.txt    # pinned deps (diffusers commit, peft; spaces unpinned)
├── pyproject.toml      # ruff py311 / line-length 120 + pytest -m 'not gpu'
├── setup.sh            # venv bootstrap (python3.11 -m venv .venv + pip install)
└── tests/              # L1 structural + L2 mocked unit; GPU smoke under -m gpu
```

## Tech stack

- **[Gradio 5.50](https://gradio.app/)** — UI shell, native components, `gr.Progress(track_tqdm=True)`
- **[diffusers](https://github.com/huggingface/diffusers) `QwenImageEditPlusPipeline`** — pipeline class + PEFT adapter wiring
- **[Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)** by the Qwen team (Apache-2.0)
- **[lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)** — Lightning LoRA for 4-step fast mode
- **HF Spaces ZeroGPU** (A10G) — `@spaces.GPU(duration=…)` with adaptive duration estimation

## Design

Theme: **Soft Dark Restraint** — warm dark substrate `#1A1614`, cream ink `#F0E8DD`, one accent `#7C5CFF` (Qwen violet) used sparingly: live status dot, slider fill, primary button, brand period. Inter throughout. No display fonts, no shadows, no gradients beyond the faint violet CTA wash. The generated image stays the visual focus.

Progressive disclosure keeps the form short:

- **Fast / Quality** toggle → reveals Negative Prompt + CFG slider only in Quality mode (no-ops in Fast)
- **Advanced** accordion → Steps and Seed, collapsed by default
- Compose tab → reference image slots (up to two) appear alongside the target

Spec and design rationale live under `docs/superpowers/`.

## Notes on running

- **57.7 GB cold start.** The full model at bfloat16 (transformer + Qwen2.5-VL-7B text encoder + VAE) is ~57.7 GB on disk. `preload_from_hub` brings weights to the Space at build time so first inference doesn't stall on a download — but pipeline construction still takes ~10–20 s on a fresh ZeroGPU slot.
- **ZeroGPU duration cap.** Fast mode reserves 60 s; Quality mode is estimated at `30 + steps × 3.5` seconds, clamped to `[60, 180]`. If a slot aborts, the backend retries once at 2× the estimated duration. The duration field is a queue-priority signal, not a billing cap.
- **No-CFG fast path.** When `true_cfg_scale = 1.0` the pipeline skips the unconditional forward pass entirely — the Lightning LoRA was distilled without CFG so this is the correct and intended setting. Forcing `true_cfg_scale > 1` in Fast mode would double runtime without improving quality.
- **Local VRAM.** Peak active VRAM is ~35–40 GB. On GPUs with less than 40 GB free, `should_cpu_offload()` returns `True` and the backend calls `pipe.enable_model_cpu_offload()` to move idle components to RAM between diffusion steps.

## License

MIT for the app code (see `LICENSE`). `Qwen/Qwen-Image-Edit-2511` is Apache-2.0. The Lightning LoRA (`lightx2v/Qwen-Image-Edit-2511-Lightning`) retains its upstream license. The diffusers library is Apache-2.0.

## Credits

`Qwen/Qwen-Image-Edit-2511` by the [Qwen team](https://github.com/QwenLM) at Alibaba. [Hugging Face diffusers](https://github.com/huggingface/diffusers) for the `QwenImageEditPlusPipeline` integration. Lightning LoRA by [lightx2v](https://huggingface.co/lightx2v). Built by [Mayank Gupta](https://huggingface.co/techfreakworm) — drop a like on the [Space](https://huggingface.co/spaces/techfreakworm/qwen-image-editor) if it's useful.
