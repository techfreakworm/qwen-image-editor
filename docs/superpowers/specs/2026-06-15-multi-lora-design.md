# Multi-LoRA Support — Design Spec

> Date: 2026-06-15 · Status: proposed (pending operator approval)
> Refs: `docs/superpowers/specs/2026-06-15-qwen-image-editor-design.md` (base design)

---

## 1. Overview

This document specifies multi-LoRA stacking for **Quality mode** on the Qwen Image Editor. The Lightning LoRA (`lightx2v/Qwen-Image-Edit-2511-Lightning`) remains Fast-mode-only and is never combined with Quality LoRAs. In Quality mode, users may enable 0–N LoRAs from a curated built-in gallery, optionally augmented by a user-supplied custom LoRA, each with an independent weight slider. The feature applies identically to both Edit and Compose tabs (with one compose-specific nuance documented in §4).

---

## 2. LoRA Stacking — Technical Approach

### 2.1 Diffusers multi-adapter API

`QwenImageEditPlusPipeline` inherits from `FluxPipeline` / `DiffusionPipeline` and carries the `LoraLoaderMixin` mixin. The public multi-adapter API is:

```python
# Load each adapter (once, at startup or lazily):
pipe.load_lora_weights(repo_id, weight_name=filename, adapter_name=unique_name)

# Before each inference call — activate all desired adapters with per-LoRA weight:
pipe.set_adapters([name_a, name_b, ...], [weight_a, weight_b, ...])

# For Quality mode with no LoRAs active:
pipe.disable_lora()

# For Fast mode (Lightning only):
pipe.set_adapters([LORA_ADAPTER_NAME], [1.0])
```

`set_adapters` replaces the active adapter set atomically — no separate enable/disable calls are needed. Weights are float scalers applied as `lora_scale` per adapter during the attention forward pass; they do **not** modify stored weights, so switching between requests is free and safe.

### 2.2 Coexistence with the Lightning adapter

The Lightning adapter (`adapter_name="lightning"`) is loaded at startup and stays resident in PEFT's adapter registry throughout the process lifetime. The critical invariant:

- **Fast mode:** `set_adapters(["lightning"], [1.0])` — only Lightning is active.
- **Quality mode, no Quality LoRAs:** `disable_lora()` — no adapters active.
- **Quality mode, with Quality LoRAs:** `set_adapters([qa, qb, ...], [wa, wb, ...])` — Lightning is **not** in this list and therefore **not** active.

Lightning is **never** included in a Quality-mode `set_adapters` call. This is enforced by construction: `_apply_speed` in `modes.py` will be extended to accept an optional `quality_adapters: list[tuple[str, float]]` parameter; when speed is "Quality", it builds the adapter list exclusively from that parameter (never touching "lightning").

### 2.3 Fuse vs. dynamic — decision: always dynamic

Fusing (`fuse_lora`) bakes adapter weights into the base model in-place. This is irreversible without `unfuse_lora`, and `unfuse_lora` is approximate (accumulates floating-point error). On ZeroGPU, the pipeline object is shared across requests and must be reusable; fusing one user's LoRA selection would corrupt the next user's run. **Always use dynamic (non-fused) `set_adapters`.**

### 2.4 Adapter lifecycle — load-once, lazy, cached

Loading a LoRA from HuggingFace Hub takes 1–5 seconds (download + PEFT registration). To avoid per-request reload churn:

1. **Built-in LoRAs** are loaded lazily on first use and cached in PEFT's adapter registry for the process lifetime. Since ZeroGPU recycles the GPU context between requests but keeps CPU memory, the PEFT registry survives. Re-loading is skipped if `adapter_name` is already registered (checked via `pipe.get_list_adapters()`).

2. **Custom LoRAs** uploaded by the user are also loaded lazily and cached under a deterministic key derived from the content hash of the file (for uploads) or `repo_id + weight_name` (for HF pull). The cache is bounded: a simple LRU of 4 custom adapters evicts the least recently used when full.

3. **Memory implications on ZeroGPU (H200, ~80 GB HBM):** Each Qwen-Image-Edit-2511 LoRA is ~50–200 MB on disk / VRAM. With 5 built-ins loaded simultaneously, the overhead is <1 GB — negligible relative to the ~20 GB base model. The dynamic (non-fused) path stores adapter delta weights alongside the base model; no duplication occurs.

### 2.5 Extended `_apply_speed` signature (pseudo-code, not implementation)

```python
# modes.py — proposed signature change
def _apply_speed(
    pipe: Any,
    speed: str,
    quality_adapters: list[tuple[str, float]] | None = None,
) -> None:
    if speed == "Fast":
        pipe.scheduler = pipe._qie_lightning_scheduler
        pipe.set_adapters([models.LORA_ADAPTER_NAME], [1.0])
    else:
        pipe.scheduler = pipe._qie_default_scheduler
        if quality_adapters:
            names = [a[0] for a in quality_adapters]
            weights = [a[1] for a in quality_adapters]
            pipe.set_adapters(names, weights)
        else:
            pipe.disable_lora()
```

`quality_adapters` is built by the backend from the LoRA gallery state passed in `params`.

---

## 3. Curated Built-in LoRA Gallery

All proposed built-ins use `Qwen/Qwen-Image-Edit-2511` as their base model and are released under Apache-2.0.

### 3.1 Proposed gallery (6 entries)

| # | Display Name | Repo ID | Filename | Trigger phrase | Rec. weight | What it does | License |
|---|---|---|---|---|---|---|---|
| 1 | **Multi-Angle** | `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` | `qwen-image-edit-2511-multiple-angles-lora.safetensors` | `<sks> [azimuth] [elevation] [distance]` — e.g. `<sks> 90 0 1.5` | 0.9 | Re-renders subject from a precise camera angle across 96 poses (4 elevations × 8 azimuths × 3 distances); trained on 3,000+ Gaussian Splatting renders | Apache-2.0 |
| 2 | **Anime Style** | `prithivMLmods/Qwen-Image-Edit-2511-Anime` | *(single safetensors in repo)* | `Transform into anime.` | 0.85 | Converts photo to anime flat-cel shading while preserving pose, proportions, and camera angle | Apache-2.0 |
| 3 | **Hyper-Realistic Portrait** | `prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait` | `HRP_20.safetensors` | `Transform into a hyper-realistic face portrait` | 0.8 | Photorealistic portrait with pore-level textures, cool-toned lighting, shallow DoF | Apache-2.0 |
| 4 | **Any-Pose Transfer** | `lilylilith/AnyPose` | *(base safetensors in repo)* | Natural-language pose description; see trigger note* | 0.7 (base) + 0.7 (helper) | Transfers any pose from a reference image to the subject without ControlNet | Apache-2.0 |
| 5 | **ICEdit (In-Context Edit)** | `DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA` | `model.safetensors` | None required; transformation is driven by example image pair | 0.85 | Three-image prompting: show before→after pair, apply same transformation to third image | Apache-2.0 |
| 6 | **Upscale 2K** | `starsfriday/Qwen-Image-Edit-2511-Upscale2K` | `qwen_image_edit_2511_upscale.safetensors` | `Upscale this picture to 4K resolution.` (no fixed trigger; any upscale instruction works) | 0.9 | Lossless enlargement to ~2K; structure-preserving super-resolution | Apache-2.0 |

*AnyPose trigger note: recommended prompt template is `"Make the person in image 1 do the exact same pose of the person in image 2. Changing the style and background is undesirable. The new pose should be pixel accurate..."` — the UI should prepend this template when AnyPose is active and the user has not included pose language themselves.

### 3.2 Licensing notes

All 6 proposed adapters are Apache-2.0. No license flags. The base model `Qwen/Qwen-Image-Edit-2511` is also Apache-2.0, making the entire stack permissively licensed.

### 3.3 Candidates considered but not included

- **Manga Tone / Noir Comic** (`prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel`): interesting but highly niche; kept as a future addition. License: Apache-2.0.
- **WaveSpeedAI LoRA**: proprietary / non-HF-hosted — excluded.
- **Lightning LoRA** (`lightx2v/Qwen-Image-Edit-2511-Lightning`): already loaded as the Fast-mode adapter; not exposed in the Quality gallery.

---

## 4. Custom LoRA Uploader / Picker

### 4.1 Two input paths

**Path A — HuggingFace repo pull**
- User enters a repo ID (e.g. `someuser/my-lora`) and optionally a filename (`my_lora.safetensors`).
- Backend calls `pipe.load_lora_weights(repo_id, weight_name=weight_name or None, adapter_name=cache_key)`.
- If `weight_name` is omitted, diffusers auto-selects the single `.safetensors` file in the repo; if multiple exist, we surface an error asking the user to specify the filename.

**Path B — File upload**
- User uploads a `.safetensors` file directly (via Gradio `gr.File`).
- File is saved to a temp path; backend calls `pipe.load_lora_weights(temp_path, adapter_name=sha256_of_file[:8])`.

### 4.2 Validation

1. **Extension gate:** reject anything that is not `.safetensors` before attempting to load (Gradio `file_types=[".safetensors"]` on the upload widget).
2. **Load-time catch:** `pipe.load_lora_weights` raises if the file is corrupt, incompatible architecture, or missing expected keys. Wrap in `try/except` and surface a user-facing error: `"LoRA could not be loaded — it may be incompatible with Qwen-Image-Edit-2511 (wrong architecture or corrupt file)."`.
3. **Compatibility signal (best-effort):** after loading, check that at least one PEFT adapter parameter with a `qwen` or `transformer_blocks` namespace was registered. If none, warn the user that the LoRA may not affect this model. This is heuristic — do not hard-block.
4. **Weight slider range:** custom LoRA weight slider is 0.0–2.0 (default 0.8). Range extends to 2.0 to accommodate LoRAs trained at high scale, unlike built-ins which cap at 1.2.

### 4.3 Security considerations

- **Arbitrary downloads (Path A):** any HF repo ID can be passed. Mitigations: (a) diffusers uses `safetensors` format which executes no Python code, unlike legacy `.pt` pickle files — no arbitrary code execution risk; (b) on ZeroGPU Spaces the process is sandboxed; (c) HuggingFace Hub downloads go through the Hub CDN, not arbitrary URLs. No additional URL-allow-list is needed on ZeroGPU.
- **File uploads (Path B):** files are validated by extension and then by the `safetensors` library's header parser before any tensor is loaded. The `safetensors` format is deliberately sandboxed. Temp files are cleaned up after the session.
- **Recommended note in UI:** "Only load LoRAs trained on `Qwen/Qwen-Image-Edit-2511`. LoRAs from other base models will produce incorrect results."

### 4.4 Custom LoRA cache

```
_custom_lora_cache: OrderedDict[str, str]  # cache_key -> adapter_name in PEFT
MAX_CUSTOM_LORAS = 4  # LRU eviction
```

On eviction, call `pipe.delete_adapters([evicted_adapter_name])` to free PEFT state. (PEFT's `delete_adapters` does not free GPU memory immediately — a subsequent inference call will GC unused delta weights.)

---

## 5. Edit vs. Compose Mode Behavior

### 5.1 Pipeline unification

Both `call_edit` and `call_compose` ultimately call `_run`, which calls `_apply_speed`. The LoRA stack is set once in `_apply_speed` before the pipeline call; neither the edit nor compose code path diverges at the LoRA layer. Stacked LoRAs therefore apply identically to both modes.

### 5.2 Per-LoRA suitability by mode

| LoRA | Edit | Compose | Notes |
|---|---|---|---|
| Multi-Angle | ✅ primary use case | ✅ useful | Works on the first (target) image |
| Anime Style | ✅ | ✅ | Applies to the composed output |
| Hyper-Realistic Portrait | ✅ | ⚠️ marginal | Useful only if the compose output is a portrait; may conflict with reference images that are non-photorealistic |
| AnyPose | ✅ | ✅ primary use case | In Compose, image 2 naturally becomes the pose reference — synergistic |
| ICEdit | ⚠️ | ✅ primary use case | ICEdit is designed for 3-image prompting, which maps directly to Compose's 3-slot layout; in single-image Edit it still works but the "before→after pair" must be embedded in the prompt |
| Upscale 2K | ✅ | ✅ | Output-level; mode-agnostic |
| Custom | ✅ | ✅ | No restriction; user's responsibility |

UI annotation: in the gallery cards, each built-in LoRA has a small "Best in: Edit / Compose / Both" badge for discoverability, but no hard lock-out. Users can use any LoRA in either mode.

### 5.3 Fast vs. Quality with multi-LoRA

**Multi-LoRA is Quality-only.** Rationale:

1. The Lightning LoRA was distilled specifically for 4-step inference with `shift=ln(3)` exponential time shifting. Stacking a style/pose LoRA on top of a distillation LoRA is untested and likely to produce degraded outputs — the combined deltas were not co-trained.
2. The Lightning LoRA's `set_adapters(["lightning"], [1.0])` call in Fast mode makes it the sole active adapter. Introducing additional adapters would require retuning the Lightning scale, which is outside the scope of this feature.
3. In Fast mode the user gets a 4-step run (~10–15 s); style LoRAs may need 20–40 steps to fully express. Quality mode's default 40 steps gives LoRAs adequate budget.

**Implementation:** the LoRA gallery and custom uploader are rendered inside the `quality_grp` Group (visible only when Quality is selected). When the user switches back to Fast, the LoRA selection state is preserved in Gradio state but not transmitted to the backend — `_apply_speed("Fast", ...)` always uses only Lightning.

---

## 6. UX Design — LoRA Picker

### 6.1 Placement

The LoRA section lives inside the existing `quality_grp` (the `gr.Group(visible=False)` that appears when Quality is selected). It is injected below `true_cfg` and above `negative_prompt`, inside a collapsible `gr.Accordion("LoRA", open=False)`.

This preserves the current tab layout — the accordion is collapsed by default so the panel does not grow on first open.

### 6.2 Component structure (Gradio)

```
gr.Accordion("LoRA", open=False)
  gr.Markdown("Quality-mode style adapters — stack up to 4. Active only in Quality mode.")
  
  # --- Built-in gallery (6 cards) ---
  gr.Row()
    [for each built-in LoRA]:
      gr.Group(elem_classes=["qie-lora-card"])
        gr.Checkbox(label="<display name>", value=False)     # toggle
        gr.Markdown("<one-line description>", visible=True)
        gr.Slider(0.0, 1.5, value=rec_weight, step=0.05,    # weight; visible only when checked
                  label="Weight", visible=False)
  
  # --- Custom LoRA accordion ---
  gr.Accordion("Custom LoRA", open=False)
    gr.Radio(["HuggingFace repo", "Upload file"], value="HuggingFace repo",
             label="Source")
    
    # Path A — HF repo
    gr.Textbox(label="Repo ID", placeholder="username/my-lora-name")
    gr.Textbox(label="Filename (optional)", placeholder="lora.safetensors")
    
    # Path B — file upload (visible when "Upload file" selected)
    gr.File(label="Upload .safetensors", file_types=[".safetensors"],
            elem_classes=["qie-lora-file"])
    
    gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Weight")
    gr.Button("Add LoRA", variant="secondary")
    gr.Markdown("", elem_id="qie-custom-lora-status")  # error/success feedback
```

The gallery cards use a `.qie-lora-card` CSS class (added to `theme.py`'s `CSS` string) for a bordered, padded tile layout. Weight sliders become visible only when their checkbox is ticked (`.change()` handler toggles `visible`).

### 6.3 ASCII wireframe — Edit tab, Quality mode expanded, LoRA accordion open

```
┌─ Edit ──────────────────────────────────────────────────────────────────────┐
│ [Input image                    ]  │  [Output image                        ] │
│                                    │                                         │
│ Prompt ────────────────────────── │  ...                                    │
│ ┌─────────────────────────────────┐ │                                        │
│ │ Make the sky dramatic…          │ │                                        │
│ └─────────────────────────────────┘ │                                        │
│                                    │                                        │
│  Speed:  ○ Fast  ● Quality         │                                        │
│                                    │                                        │
│ ┌── Quality settings ────────────┐ │                                        │
│ │ CFG scale ────────────●─────── │ │                                        │
│ │                                │ │                                        │
│ │ ▼ LoRA ────────────────────── │ │                                        │
│ │  Quality-mode adapters (≤4)    │ │                                        │
│ │                                │ │                                        │
│ │  ┌────────────┐ ┌────────────┐ │ │                                        │
│ │  │ ☑ Multi-   │ │ ☐ Anime    │ │ │                                        │
│ │  │  Angle     │ │  Style     │ │ │                                        │
│ │  │ Re-render  │ │ Cel-shad-  │ │ │                                        │
│ │  │ from any   │ │ ing, pres- │ │ │                                        │
│ │  │ viewpoint  │ │ erves pose │ │ │                                        │
│ │  │ Best: Both │ │ Best: Both │ │ │                                        │
│ │  │ Weight ──● │ │            │ │ │                                        │
│ │  │ [0.90]     │ │            │ │ │                                        │
│ │  └────────────┘ └────────────┘ │ │                                        │
│ │  ┌────────────┐ ┌────────────┐ │ │                                        │
│ │  │ ☐ Portrait │ │ ☐ AnyPose  │ │ │                                        │
│ │  │ Hyper-     │ │ Pose from  │ │ │                                        │
│ │  │ realistic  │ │ reference  │ │ │                                        │
│ │  │ face       │ │ Best:      │ │ │                                        │
│ │  │ Best: Edit │ │ Compose    │ │ │                                        │
│ │  └────────────┘ └────────────┘ │ │                                        │
│ │  ┌────────────┐ ┌────────────┐ │ │                                        │
│ │  │ ☐ ICEdit   │ │ ☐ Upscale  │ │ │                                        │
│ │  │ 3-image    │ │ 2K         │ │ │                                        │
│ │  │ example-   │ │ Lossless   │ │ │                                        │
│ │  │ driven     │ │ enlarge    │ │ │                                        │
│ │  │ Best:      │ │ Best: Both │ │ │                                        │
│ │  │ Compose    │ │            │ │ │                                        │
│ │  └────────────┘ └────────────┘ │ │                                        │
│ │                                │ │                                        │
│ │  ▶ Custom LoRA ──────────────  │ │                                        │
│ │                                │ │                                        │
│ │ Negative prompt ─────────────  │ │                                        │
│ └────────────────────────────────┘ │                                        │
│                                    │                                        │
│ ▶ Advanced ────────────────────── │                                        │
│                                    │                                        │
│ [Generate]                         │                                        │
└────────────────────────────────────┴────────────────────────────────────────┘
```

### 6.4 Visual treatment (Soft Dark Restraint)

- **Card background:** `#14110F` (input_bg) with `1px solid #2A241E` border and `6px` radius.
- **Active card (checkbox ticked):** border upgrades to `1px solid rgba(124,92,255,0.5)`, subtle violet left-border accent `3px solid #7C5CFF`, background lifts to `rgba(124,92,255,0.04)`.
- **Badge "Best in: Compose":** 10px capsule, `color: #988B7C`, `background: #1F1B17`.
- **Weight slider:** inherits `slider_color: #7C5CFF` from theme — no extra CSS needed.
- **Card grid:** `gr.Row(equal_height=True)` with 2 columns, wrapping across rows. On mobile (<600px), single column.
- **"Add LoRA" button:** `variant="secondary"` (bordered, not filled). On success, status line reads `"✓ my-lora loaded (sha: abc12345)"` in `#7C5CFF`. On error, `"✗ Could not load — …"` in `#E05555`.

### 6.5 State passed to backend

The LoRA state is serialized into `params` as:

```python
params["loras"] = [
    {"adapter_name": "fal__multi-angle", "weight": 0.9},
    {"adapter_name": "custom__abc12345", "weight": 1.1},
]
```

The backend resolves each entry: if the adapter is not yet in PEFT's registry, it loads it; then calls `set_adapters([...], [...])` with the full list. The `adapter_name` for built-ins is the repo ID with `/` replaced by `__` and a fixed suffix (e.g. `fal__Qwen-Image-Edit-2511-Multiple-Angles-LoRA`).

---

## 7. `models.py` Constants (additions, not modifications)

The following constants would be added to `models.py` to centralize built-in gallery metadata. They are read-only config, not runtime state:

```python
BUILTIN_LORAS: list[dict] = [
    {
        "display_name": "Multi-Angle",
        "adapter_name": "fal__multi-angle",
        "repo_id": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        "weight_name": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        "trigger": "<sks> [azimuth] [elevation] [distance]",
        "default_weight": 0.9,
        "description": "Re-render subject from any camera angle (96 poses).",
        "best_mode": "Both",
    },
    {
        "display_name": "Anime Style",
        "adapter_name": "prithiv__anime",
        "repo_id": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
        "weight_name": None,  # auto-detect single safetensors
        "trigger": "Transform into anime.",
        "default_weight": 0.85,
        "description": "Flat cel-shading; preserves pose and proportions.",
        "best_mode": "Both",
    },
    {
        "display_name": "Hyper-Realistic Portrait",
        "adapter_name": "prithiv__portrait",
        "repo_id": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weight_name": "HRP_20.safetensors",
        "trigger": "Transform into a hyper-realistic face portrait",
        "default_weight": 0.8,
        "description": "Pore-level textures, shallow DoF, cool-toned lighting.",
        "best_mode": "Edit",
    },
    {
        "display_name": "AnyPose",
        "adapter_name": "lilylilith__anypose",
        "repo_id": "lilylilith/AnyPose",
        "weight_name": None,
        "trigger": None,  # template injected by backend
        "default_weight": 0.7,
        "description": "Transfer any pose from a reference image.",
        "best_mode": "Compose",
    },
    {
        "display_name": "ICEdit",
        "adapter_name": "diffsynth__icedit",
        "repo_id": "DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA",
        "weight_name": "model.safetensors",
        "trigger": None,  # example-pair driven
        "default_weight": 0.85,
        "description": "Apply transformations from an example before→after pair.",
        "best_mode": "Compose",
    },
    {
        "display_name": "Upscale 2K",
        "adapter_name": "starsfriday__upscale2k",
        "repo_id": "starsfriday/Qwen-Image-Edit-2511-Upscale2K",
        "weight_name": "qwen_image_edit_2511_upscale.safetensors",
        "trigger": "Upscale this picture to 4K resolution.",
        "default_weight": 0.9,
        "description": "Lossless upscale to ~2K; structure-preserving.",
        "best_mode": "Both",
    },
]
```

---

## 8. Risks and Open Questions

| Risk | Severity | Mitigation |
|---|---|---|
| AnyPose has separate "base" and "helper" adapter files — load_lora_weights may need two calls | Medium | Research AnyPose repo file list; may require `adapter_name="anypose_base"` + `"anypose_helper"` with separate sliders or a single averaged weight |
| Qwen anime LoRA filename not confirmed (auto-detect path) | Low | diffusers auto-selects if exactly one `.safetensors` exists; add explicit check at load time |
| ZeroGPU cold-start latency increases if multiple LoRAs load on first request | Medium | Pre-warm: load all built-in LoRAs during `_build_pipeline()` (pipeline construction), not lazily — trading startup time for predictable per-request latency |
| set_adapters with empty list behavior | Low | Always call `disable_lora()` for the empty-list case; never pass `set_adapters([], [])` |
| LoRA stacking interactions between style + pose + upscale | Medium | UX: warn when >2 LoRAs are active; order weight sliders with a note "effects may interact unpredictably" |
| Custom LoRA architecture incompatibility (wrong base model) | Medium | Load-time catch + heuristic namespace check (§4.2); surface clear error |

---

## 9. Out of Scope for This Spec

- Training new LoRAs via the UI (no fine-tuning surface).
- LoRA merging / averaging (separate feature).
- Exposing LoRAs in Fast mode (architecturally incompatible with Lightning distillation — explicitly excluded).
- Inpainting/ControlNet integration (excluded from base design §2).
