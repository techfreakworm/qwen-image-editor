# Qwen Image Editor — Implementation Plan

> **For agentic workers:** Build each file by MIRRORING the equivalent file in `.refs/z-image-studio/` and applying the Qwen deltas below. The verbatim reference patterns (ZeroGPU wiring, theme, CSS, test tricks) are in `.refs/research/mirror-blueprint.md`; the exact diffusers/ZeroGPU/Lightning recipe is in `.refs/research/diffusers-zerogpu-api.md`. Steps use `- [ ]` checkboxes.

**Goal:** A Gradio app wrapping `Qwen/Qwen-Image-Edit-2511` (Edit + Compose tabs, Fast/Quality toggle) that runs on local CUDA and HF ZeroGPU.

**Architecture:** Flat module layout mirroring z-image-studio. One `QwenImageEditPlusPipeline`; Fast/Quality via PEFT adapter enable/disable + scheduler swap. Heavy imports deferred so GPU-free CI passes.

**Tech Stack:** Python 3.11, Gradio 5.50.0, diffusers (pinned commit), torch bfloat16, peft, pytest (`-m 'not gpu'`), ruff.

---

## INTERFACE CONTRACT (all modules build to this — prevents drift)

### `params` dict (UI → `backend.generate(mode, params)` → mode handler)
```
mode:            "edit" | "compose"
prompt:          str
images:          list[PIL.Image.Image]   # edit: [target]; compose: [target, ref1?, ref2?] (1..3)
speed:           "Fast" | "Quality"
steps:           int                      # default 4 (Fast) / 40 (Quality)
true_cfg:        float                    # default 1.0 (Fast) / 4.0 (Quality)
negative_prompt: str                      # default " "
seed:            int                       # -1 => random
```

### `models.py` public API
```python
MODEL_ID  = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LORA_ADAPTER_NAME = "lightning"
LIGHTNING_SCHEDULER_CONFIG: dict   # exact dict from diffusers-zerogpu-api.md §2 (base_shift=math.log(3), time_shift_type="exponential", use_dynamic_shifting=True, ...)

def on_spaces() -> bool                      # bool(os.environ.get("SPACES_ZERO_GPU"))
def auto_device() -> str                     # cuda > mps > cpu (lazy import torch)
def fit_dimensions(image, max_pixels=1024*1024, multiple=16) -> tuple[int,int]  # preserve aspect, area<=max_pixels, w/h rounded to `multiple`, min 256
def should_cpu_offload(device: str) -> bool  # True iff local cuda AND free VRAM < 40 GB; False on spaces/mps/cpu (lazy import torch)
```

### `modes.py` public API
```python
def call_edit(pipe, params: dict) -> tuple[PIL.Image.Image, dict]
def call_compose(pipe, params: dict) -> tuple[PIL.Image.Image, dict]
DISPATCH = {"edit": call_edit, "compose": call_compose}
# internal: _apply_speed(pipe, speed) ; _run(pipe, params)
# _run: validate images non-empty (else ValueError); _apply_speed; (w,h)=fit_dimensions(images[0]);
#       seed = randint if params["seed"]<0 else params["seed"]; gen=torch.Generator(pipe.device or "cpu").manual_seed(seed)
#       out = pipe(image=images, prompt=..., negative_prompt=..., true_cfg_scale=..., num_inference_steps=..., height=h, width=w, generator=gen).images[0]
#       meta = {mode, speed, steps, true_cfg, seed, width, height, num_inputs:len(images)}
# _apply_speed Fast:    pipe.scheduler = pipe._qie_lightning_scheduler ; pipe.set_adapters([LORA_ADAPTER_NAME], [1.0])
# _apply_speed Quality: pipe.scheduler = pipe._qie_default_scheduler   ; pipe.disable_lora()
```

### `backend.py` public API
```python
def duration_for(mode: str, params: dict) -> int   # Fast->~60 ; Quality-> 30 + steps*3.5, clamp [60,180]
class QwenImageEditBackend:
    def __init__(self): self.pipeline = _build_pipeline()
    @_GPU
    def generate(self, mode: str, params: dict) -> tuple[PIL.Image.Image, dict]   # DISPATCH[mode](self.pipeline, params)
def generate_with_retry(backend, mode, params)     # retry once on "gpu task aborted" (no behavior change locally)
# _build_pipeline(): lazy import torch, diffusers; build default pipe; store pipe._qie_default_scheduler=pipe.scheduler;
#   build Lightning scheduler from models.LIGHTNING_SCHEDULER_CONFIG -> pipe._qie_lightning_scheduler;
#   pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, adapter_name=LORA_ADAPTER_NAME);
#   device placement (on_spaces->.to("cuda"); should_cpu_offload->enable_model_cpu_offload(); else .to(device));
#   start in Fast state; return pipe.
# _GPU = spaces.GPU(duration=lambda *a, **kw: duration_for(*a[1:3])) if (spaces and _ON_SPACES) else _identity
```

### `ui.py` component-dict keys (per tab builder)
- Common: `prompt, speed, steps, true_cfg, negative_prompt, seed, generate_btn, output_image, output_meta, advanced_grp, quality_grp`
- Edit tab adds: `image`
- Compose tab adds: `target_image, ref_image_1, ref_image_2`
- `speed` = `gr.Radio(["Fast","Quality"], value="Fast")`; `quality_grp` (negative_prompt + true_cfg) starts `visible=False`; `advanced_grp` = `gr.Accordion("Advanced", open=False)` holding steps/seed.

### `app.py` behaviors
- `_get_backend()` lazy singleton (build on first generate).
- `on_edit_generate(...)` / `on_compose_generate(...)`: assemble `params`, drop None images from the list, call `generate_with_retry`, return `(image, meta)`. `gr.Progress(track_tqdm=True)` param present.
- `speed.change` -> set steps/true_cfg defaults (Fast:4/1.0, Quality:40/4.0) and `gr.Group(visible = speed=="Quality")` for `quality_grp`.
- Env at top of file BEFORE imports: `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")`.

### `tooltips.py` keys
`prompt, image, target_image, ref_image, speed, steps, true_cfg, negative_prompt, seed` (each non-empty, <=200 chars).

### `theme.py`
`PALETTE` = Soft Dark Restraint but `accent="#7C5CFF"`, `accent_text="#0E0A1A"`; primary_hue = violet 11-stop scale centered on `#7C5CFF`; neutral hue unchanged from blueprint §5; CSS identical to blueprint §5 with `zis-`→`qie-` and amber→violet.

---

## Task 1: Scaffold + meta files
**Files:** Create `LICENSE` (MIT, "Mayank Gupta", 2026), `pyproject.toml` (verbatim blueprint §7, ruff py311/line 120 + pytest `-m 'not gpu'` + gpu marker), `requirements.txt` (spec deps; comment why `spaces` unpinned, why diffusers pinned to commit, peft included), `setup.sh` (verbatim blueprint §9), `.github/workflows/ci.yml` (blueprint §9 — install only ruff/pytest/pytest-mock/pillow/numpy/gradio==5.50.0/safetensors), `tests/__init__.py` (empty), `tests/conftest.py` (sys.path insert, blueprint §10).
- [ ] Write all meta files.
- [ ] Write `tests/test_scaffold.py`: assert each source/meta file exists; pyproject has `py311` + `-m 'not gpu'`; requirements has gradio/diffusers/peft and NOT a pinned `spaces`; LICENSE is MIT with author "Mayank Gupta".
- [ ] `ruff format . && ruff check .` clean.

## Task 2: theme.py + tooltips.py
**Files:** Create `theme.py`, `tooltips.py`, `tests/test_theme.py`, `tests/test_tooltips.py`.
- [ ] `theme.py`: `PALETTE`, `build_theme()`, `CSS` per the theme.py contract above (violet accent, `qie-` CSS prefix).
- [ ] `tooltips.py`: `TOOLTIPS` with all keys above.
- [ ] `test_theme.py`: PALETTE accent == "#7C5CFF"; `build_theme()` returns `gr.themes.Base`; CSS contains `.qie-` selectors and no `.zis-`/amber `#FFB02E`.
- [ ] `test_tooltips.py`: all keys present, values non-empty str, <=200 chars.
- [ ] ruff clean.

## Task 3: models.py
**Files:** Create `models.py`, `tests/test_models.py`.
- [ ] Implement the `models.py` contract (constants, `LIGHTNING_SCHEDULER_CONFIG` verbatim from api ref §2, `on_spaces`, `auto_device`, `fit_dimensions`, `should_cpu_offload`). Heavy imports (torch) deferred inside functions.
- [ ] `test_models.py`: `MODEL_ID/LORA_REPO/LORA_FILE` correct; `on_spaces()` reads env; `fit_dimensions` returns multiples of 16, area<=max, preserves aspect within tolerance, min 256; `LIGHTNING_SCHEDULER_CONFIG["time_shift_type"]=="exponential"`. `auto_device` test `skipif` torch missing.
- [ ] ruff + `pytest tests/test_models.py` green (GPU-free).

## Task 4: modes.py
**Files:** Create `modes.py`, `tests/test_modes.py`.
- [ ] Implement `call_edit`, `call_compose`, `_run`, `_apply_speed`, `DISPATCH` per contract. Lazy import torch for the generator.
- [ ] `test_modes.py` (MagicMock pipe, `pipe(...)` returns `MagicMock(images=[Image.new(...)])`): edit passes exactly the target image; compose passes up to 3 images dropping Nones; Fast => set_adapters called + steps 4 + true_cfg 1.0 in the pipe call kwargs; Quality => disable_lora called + steps 40 + true_cfg 4.0; seed=-1 produces an int seed in meta; empty images raises ValueError; meta has the documented keys.
- [ ] ruff + pytest green.

## Task 5: backend.py
**Files:** Create `backend.py`, `tests/test_backend.py`.
- [ ] Implement `import spaces` try/except, `_ON_SPACES`, `_identity`, `duration_for`, `_GPU`, `QwenImageEditBackend`, `_build_pipeline`, `generate_with_retry` per contract + blueprint §3. `_build_pipeline` lazy-imports torch + `from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler` and follows api ref §2/§4/§5.
- [ ] `test_backend.py` (monkeypatch `backend._build_pipeline` -> MagicMock): `duration_for` clamps to [60,180]; Quality duration > Fast; `generate("edit", params)` routes to `modes.DISPATCH["edit"]` (monkeypatch DISPATCH); `generate_with_retry` retries exactly once on an exception containing "gpu task aborted" and re-raises other errors.
- [ ] ruff + pytest green.

## Task 6: ui.py
**Files:** Create `ui.py`, `tests/test_ui.py`.
- [ ] `build_edit_tab()` / `build_compose_tab()` returning the documented dicts; progressive disclosure (advanced accordion, quality_grp hidden); `info=TOOLTIPS[...]`; left controls / right output columns per blueprint §6. No event wiring.
- [ ] `test_ui.py` (autouse `gr.Blocks()` ctx fixture): both builders return dicts with the required keys; `speed` choices == ["Fast","Quality"] default "Fast"; `quality_grp.visible is False`; compose has `target_image/ref_image_1/ref_image_2`, edit has `image`.
- [ ] ruff + pytest green.

## Task 7: app.py
**Files:** Create `app.py`, `tests/test_app.py`.
- [ ] Top-of-file env setdefault; imports (gradio, backend, models, theme, ui, tooltips); `_get_backend()` lazy singleton; `on_edit_generate` / `on_compose_generate`; speed-change helper `_speed_defaults(speed)->(steps,true_cfg, gr.update(visible=...))`; `build_app()` (Blocks(theme,css,title), header HTML, CTA HTML, Tabs[Edit,Compose], wire `.click()` + `speed.change`); `if __name__=="__main__": build_app().queue().launch()`.
- [ ] `test_app.py`: `_speed_defaults("Fast")==(4,1.0,...)`, `("Quality")==(40,4.0,...)`; `build_app()` returns a `gr.Blocks` without building the backend (singleton not instantiated at build time).
- [ ] ruff + pytest green.

## Task 8: README.md
**Files:** Create `README.md`.
- [ ] HF Space YAML frontmatter: `title: Qwen Image Editor`, distinct emoji, `colorFrom/To` violet-ish, `sdk: gradio`, `sdk_version: "5.50.0"`, `app_file: app.py`, `python_version: "3.11"`, `suggested_hardware: zero-a10g` (note: will set ZeroGPU on the Space), `hf_oauth: false`, and `preload_from_hub` for `Qwen/Qwen-Image-Edit-2511` (transformer/vae/text_encoder/tokenizer/scheduler/processor globs) + the Lightning LoRA file (api ref §5).
- [ ] Body per blueprint §8 section order: title+badges, live demo link, What's inside (Edit/Compose + Fast/Quality table), Quick start (local) `setup.sh`, Quick start (HF Spaces), Architecture, Project layout, Tech stack, Design, Notes on running (cold start, ZeroGPU duration), License (MIT), Credits (Qwen, diffusers, lightx2v, built by Mayank Gupta).
- [ ] Update `test_scaffold.py` to assert README frontmatter has `sdk: gradio` + `preload_from_hub` mentions `Qwen/Qwen-Image-Edit-2511`.

## Task 9: Local quality gate (no GPU)
- [ ] `ruff format --check . && ruff check .` clean.
- [ ] `pytest -q` green (all `-m 'not gpu'` tests).
- [ ] Import smoke: in a venv with gradio (no torch) `python -c "import app; app.build_app()"` succeeds (proves deferred-import discipline + GPU-free UI build).
- [ ] Commit logically (scaffold, modules, tests, docs).

## Task 10: Deploy (pre-approved: Space + backing repo only)
- [ ] Create GitHub repo `techfreakworm/qwen-image-editor` (public), push `main`.
- [ ] Create HF Space `techfreakworm/qwen-image-editor` (gradio, ZeroGPU hardware), add HF remote, push.
- [ ] Watch Space build (preload 57.7 GB) → RUNNING.
- [ ] L4 UI test on the Space: Edit (Fast+Quality), Compose with 2 refs (Fast+Quality), seed reproducibility, error on missing image. Fix-loop until every path is green.

## Self-review notes
- Spec coverage: every spec capability/module maps to a task (1-8 build, 9 gate, 10 deploy/L4 test). ✓
- Interface consistency: `params` keys, `DISPATCH`, component-dict keys, and `_apply_speed`/`duration_for` signatures are defined once above and referenced by Tasks 4-7. ✓
- No placeholders: per-file deltas + the two `.refs/research/*.md` references supply concrete code; constants and signatures are explicit. ✓
