"""Pre-inference memory budgeting + OOM-avoidance planning for Apple Silicon (MPS).

Implements the qwen-brain ruling for the never-OOM mandate:

* **Dynamic budget** keyed off LIVE available unified memory (recomputed every
  preflight), NOT a fixed fraction of the 128 GB total — the box is shared, so we
  must react to what other sessions are actually using right now.
* **Two gates**: a LOAD gate (can the ~58 GB resident weights fit *before* we build
  the pipeline?) and an INFER gate (does resident + this request's activation peak
  fit?), evaluated per request.
* **Soft auto-degrade with a hard floor**: walk a degrade ladder (resolution → refs
  → drop CFG double-pass → Quality→Fast); if even the bottom rung won't fit, refuse
  with an actionable message rather than OOM/​swap-thrash.

On unified memory, ``enable_model_cpu_offload`` saves no physical RAM (CPU+GPU share
one pool), so it is deliberately NOT a degrade rung here.

All heavy imports (torch, psutil) are deferred so this module imports GPU-free in CI.
"""

from __future__ import annotations

import os

GB = 1024**3

# Resident bf16 weights — measured from the HF repo metadata:
#   transformer 40.86 + text_encoder (Qwen2.5-VL-7B) 16.58 + vae 0.25 = 57.7 GB,
#   plus ~0.25 GB when the VAE is upcast to fp32 on MPS (anti black-image).
RESIDENT_GB = 58.0
LORA_GB = 1.0  # Lightning adapter is always loaded (toggled, not unloaded)
LOAD_TRANSIENT_GB = 5.0  # shard materialization transient during from_pretrained
DEFAULT_RESERVE_GB = 20.0  # OS + other sessions + macOS 'available' is an estimate (MPS, shared RAM)
DEFAULT_CUDA_RESERVE_GB = 3.0  # CUDA context + allocator fragmentation (dedicated VRAM, not shared)

# --- Activation model (recalibrated from MEASURED MPS peaks, fresh-process) -------------
# Measured driver_allocated peaks @ 1024^2 output: edit/Fast n_img=1 -> 28.1, edit/Quality
# n_img=1 -> 28.1, compose/Fast n_img=2 -> 40.8, compose/Quality n_img=2 -> 42.7 GB.
# Findings: ~14 GB per INPUT image dominates (the pipeline auto-resizes inputs to fixed
# areas — CONDITION 384^2 + VAE 1024^2 — so this per-image encode cost is ~CONSTANT in the
# OUTPUT resolution); CFG double-pass adds only ~0-2 GB; no super-linear (n=1->2 is linear).
# Coefficients are an UPPER ENVELOPE over the measured points (conservative -> never-OOM).
# _ACT_BASE carries a fragmentation margin: live long-lived peaks ran ~2 GB above the
# fresh-process fit (compose/Fast measured 40.8 fresh but 45.6 in the live app), so the
# base is raised from the bare fresh-fit (~15) to cover the fragmented regime on the first
# run; calibration then ratchets up further for any config that exceeds it.
_ACT_BASE = 18.0  # fixed overhead (Metal/kernel caches, fragmentation, base buffers)
_ACT_PER_IMG = 13.5  # GB per input image (VL 384^2 + VAE 1MP encode + input latent tokens)
_ACT_DENOISE = 0.5  # GB per (n_img+1)*MP_out*cfg_fac (output denoise + VAE decode; small)

# CUDA coefficients are ~8x LOWER than MPS: on CUDA the model uses FlashAttention/efficient
# SDPA, which does NOT materialize the large attention tensors that MPS does. MEASURED on
# ZeroGPU (RTX Pro 6000): edit/Fast n_img=1 @1024^2 peak ABOVE the resident baseline = 4.2 GB
# (vs the MPS formula's ~32 GB). These are conservative (~3x the measurement) so the first
# run of every Edit/Compose/Quality config fits the ~37 GB free without a FALSE degrade;
# the device-keyed calibration (trust-calib on CUDA) then refines each config to its real
# ~4-10 GB, and the catchable-CUDA-OOM retry remains the hard never-OOM backstop.
_ACT_BASE_CUDA = 8.0
_ACT_PER_IMG_CUDA = 6.0
_ACT_DENOISE_CUDA = 0.6

# Resolution rungs (max_pixels) for the degrade ladder — the dominant lever.
RES_RUNGS = (1024 * 1024, 896 * 896, 768 * 768, 640 * 640)

# In-process calibration cache: (mode, res_bucket, n_ref, quality) -> measured peak GB.
# Populated by record_peak() after each successful run; used in place of the
# conservative formula thereafter (x safety margin already baked into the record).
_CALIB: dict[tuple, float] = {}
_CALIB_MARGIN = 1.1  # measured peak is already the high-water mark; small slack for proxy miss


def _reserve_gb() -> float:
    try:
        return float(os.environ.get("QIE_MPS_RESERVE_GB", DEFAULT_RESERVE_GB))
    except (TypeError, ValueError):
        return DEFAULT_RESERVE_GB


def _cuda_reserve_gb() -> float:
    try:
        return float(os.environ.get("QIE_CUDA_RESERVE_GB", DEFAULT_CUDA_RESERVE_GB))
    except (TypeError, ValueError):
        return DEFAULT_CUDA_RESERVE_GB


def available_gb() -> float:
    """Live available system memory (GB). macOS reports a reclaimable-inclusive estimate."""
    import psutil

    return psutil.virtual_memory().available / GB


def budget_gb(device: str) -> float:
    """Dynamic memory budget (GB). Recompute every preflight — cheap.

    BUDGET = min(0.90 * mps.recommended_max_memory, available - reserve)

    The physical (available - reserve) term binds on a loaded box; the
    recommended_max cap is a backstop for when `available` is transiently high.
    """
    phys = available_gb() - _reserve_gb()
    if device == "mps":
        import torch

        cap = 0.90 * torch.mps.recommended_max_memory() / GB
        return min(cap, phys)
    return phys  # CPU best-effort; CUDA has its own VRAM logic in models.should_cpu_offload


def _res_bucket(height: int, width: int) -> int:
    px = height * width
    for mp in RES_RUNGS:
        if px >= mp - 1:
            return mp
    return RES_RUNGS[-1]


def fit_dims(base_w: int, base_h: int, max_pixels: int, multiple: int = 16) -> tuple[int, int]:
    """Same contract as models.fit_dimensions, parameterised by max_pixels (no PIL dep)."""
    w, h = base_w, base_h
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        w = int(w * scale)
        h = int(h * scale)
    w = max(256, (w // multiple) * multiple)
    h = max(256, (h // multiple) * multiple)
    return w, h


def activation_gb(height: int, width: int, n_ref: int, cfg_on: bool, device: str = "mps") -> float:
    """Conservative (over-estimating) peak activation memory (GB) for one inference.

    Device-specific coefficients: MPS uses the big upper-envelope (it materializes attention
    tensors); CUDA uses ~8x lower coefficients (FlashAttention; measured ~4 GB for edit/Fast
    1-img). ``n_img`` = 1 + n_ref input images. The per-image encode term is ~fixed in output
    resolution (the pipeline auto-resizes inputs to fixed areas); only the denoise/decode term
    scales with output MP. ``cfg_on`` (true_cfg>1) doubles ONLY the denoise term — the VL/VAE
    conditioning is computed once and reused for the cond+uncond passes.
    """
    base, per_img, denoise = (
        (_ACT_BASE_CUDA, _ACT_PER_IMG_CUDA, _ACT_DENOISE_CUDA)
        if device == "cuda"
        else (_ACT_BASE, _ACT_PER_IMG, _ACT_DENOISE)
    )
    n_img = 1 + n_ref
    mp_out = (height * width) / (1024 * 1024)
    cfg_fac = 2 if cfg_on else 1
    return base + per_img * n_img + denoise * (n_img + 1) * mp_out * cfg_fac


def _act_estimate(device: str, height: int, width: int, n_ref: int, cfg_on: bool) -> float:
    """Activation-only estimate (GB), device-aware.

    The formula is already re-fit to measured reality (+margin), so a calibrated value
    BELOW it carries no information, only risk: a low cross-request driver-delta (e.g. the
    15.5 GB artifact when a prior run left the MPS heap warm and empty_cache didn't fully
    release it) must NEVER be allowed to drop the estimate below the real peak, or the
    preflight under-budgets and OOMs on a loaded box. So calibration (and penalize) can
    only RAISE the estimate above the formula, never lower it. record_peak still ratchets
    up for genuinely-higher fragmented peaks.
    """
    key = (device, "act", _res_bucket(height, width), n_ref, cfg_on)
    formula = activation_gb(height, width, n_ref, cfg_on, device)
    calib = _CALIB.get(key)
    if calib is None:
        return formula
    if device == "cuda":
        # CUDA has an ACCURATE peak counter (reset_peak_memory_stats + max_memory_allocated),
        # so a recorded peak REPLACES the (MPS-derived, inflated) formula — letting Compose
        # relax to CUDA reality (~25-30 GB) instead of being pinned at ~46 GB and forever
        # over-degrading on a 96 GB GPU. record_peak floors it at _ACT_BASE.
        return calib
    # MPS: UPWARD-ONLY ratchet — the driver-delta is an unreliable sticky-heap proxy, so
    # calibration can only RAISE the conservative formula, never lower it below the real peak.
    return max(formula, calib)


def activation_budget_gb(device: str) -> float:
    """Free memory available for THIS inference's activations (GB).

    KEY CORRECTNESS POINT: at inference time the ~58 GB weights are ALREADY resident,
    so psutil.available has already dropped to exclude them. The activation therefore
    only needs to fit in the *remaining* free RAM (minus reserve), and within the MPS
    working-set headroom above what is already allocated. Comparing resident+activation
    against a post-load `available` would double-count the now-resident weights and
    spuriously refuse every request (observed in the first smoke test).
    """
    if device == "cuda":
        import torch

        # Activation budget on CUDA = TOTAL VRAM - currently-ALLOCATED tensors - reserve.
        # NOT mem_get_info()'s driver-"free": after loading the ~58 GB model the PyTorch
        # caching allocator RESERVES most of VRAM, and the driver counts reserved-but-unused
        # as "used" -> free reads ~0 -> we'd spuriously refuse every request (observed:
        # "free 0 GB"). The activation reuses that reserved pool, so (total - allocated) is
        # the real headroom. e.g. xlarge 96 GB - 58 allocated - 3 reserve = ~35 GB for acts.
        _free_b, total_b = torch.cuda.mem_get_info()
        allocated_b = torch.cuda.memory_allocated()
        return max(0.0, (total_b - allocated_b) / GB - _cuda_reserve_gb())
    free = available_gb() - _reserve_gb()
    if device == "mps":
        import torch

        # Budget cap = 0.95 * recommended_max, intentionally BELOW the 1.0 allocator HIGH
        # watermark (app.py): that ~5 GB gap absorbs allocator fragmentation so a
        # legitimately-approved run doesn't spuriously trip the watermark. The heaviest
        # real mode (compose/Quality, ~42.7 GB act -> ~97.7 GB total) fits under 0.95.
        # Do NOT raise this to 1.0 — it would erase the fragmentation gap.
        cap = 0.95 * torch.mps.recommended_max_memory() / GB
        used = torch.mps.driver_allocated_memory() / GB
        return max(0.0, min(free, cap - used))
    return max(0.0, free)


def record_peak(
    device: str, mode: str, height: int, width: int, n_ref: int, cfg_on: bool, measured_peak_gb: float
) -> None:
    """Record a measured activation peak for calibration (FIX #2: floor-guarded).

    Rejects an implausibly-small delta (driver heap released before the read, or a proxy
    miss) so the cache can never be poisoned toward ~0 — which would silently disable
    activation budgeting and cause an OOM. Keeps the max (x margin) across runs, never
    below the physical floor. Cache is device-keyed (MPS driver-delta vs CUDA accurate
    peak-counter measurements must never cross-pollute).
    """
    floor = _ACT_BASE  # any real inference allocates at least the base overhead
    if measured_peak_gb < floor:
        return
    key = (device, "act", _res_bucket(height, width), n_ref, cfg_on)
    _CALIB[key] = max(_CALIB.get(key, 0.0), measured_peak_gb * _CALIB_MARGIN)


def load_gate(device: str) -> tuple[bool, float, float]:
    """Can resident weights fit BEFORE building? Returns (ok, need_gb, budget_gb).

    Resolution can't fix a resident-weights overflow, so a failure here means refuse.
    """
    need = RESIDENT_GB + LORA_GB + LOAD_TRANSIENT_GB
    budget = budget_gb(device)
    return need <= budget, need, budget


def top_consumers(n: int = 5) -> str:
    """Human-readable list of the biggest memory-using processes (for refusal messages)."""
    try:
        import psutil

        procs = []
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                procs.append((p.info["memory_info"].rss, p.info["name"]))
            except Exception:
                continue
        procs.sort(reverse=True)
        return ", ".join(f"{name}={rss / GB:.1f}GB" for rss, name in procs[:n])
    except Exception:
        return "(unavailable)"


def plan_request(
    device: str,
    mode: str,
    base_w: int,
    base_h: int,
    n_ref: int,
    speed: str,
    steps: int,
    true_cfg: float,
) -> dict:
    """Walk the degrade ladder; return the least-degraded plan that fits BUDGET.

    Returns a dict: {width,height,steps,true_cfg,n_ref,speed,refused,note,degrades,
    need_gb,budget_gb}. ``refused=True`` means even the bottom rung overflows.

    Degrade order (revised from the measured data): the per-input-image cost (~13.5 GB)
    is FIXED in output resolution (the pipeline auto-resizes inputs), so REF-DROP is the
    only strong lever and comes FIRST; CFG-drop and RESOLUTION are weak (~0-2 GB) and come
    later. Steps are NOT a memory lever (flat in memory).
    """
    act_budget = activation_budget_gb(device)
    quality = speed == "Quality"
    is_compose = mode == "compose"
    full_w, full_h = fit_dims(base_w, base_h, RES_RUNGS[0])
    min_ref = 0 if is_compose else n_ref  # edit can't shed inputs

    def candidates():
        # (mp, n_ref, true_cfg, speed) — ordered least-degraded -> most.
        yield RES_RUNGS[0], n_ref, true_cfg, speed  # full, no degrade
        if is_compose:  # drop reference images — the dominant lever, at FULL resolution
            for r in range(n_ref - 1, -1, -1):
                yield RES_RUNGS[0], r, true_cfg, speed
        if quality:  # drop CFG double-pass (weak)
            yield RES_RUNGS[0], min_ref, 1.0, speed
        for mp in RES_RUNGS[1:]:  # resolution down (weak lever) — last
            yield mp, min_ref, (1.0 if quality else true_cfg), speed
        if quality:  # force Quality->Fast (Lightning 4-step)
            yield RES_RUNGS[-1], min_ref, 1.0, "Fast"

    for mp, nref_c, cfg_c, speed_c in candidates():
        w, h = fit_dims(base_w, base_h, mp)
        act = _act_estimate(device, w, h, nref_c, cfg_c > 1.0)
        if act <= act_budget:
            degrades = []
            if nref_c < n_ref:
                degrades.append(f"inputs {n_ref + 1}->{nref_c + 1}")
            if cfg_c <= 1.0 < true_cfg:
                degrades.append("cfg double-pass off")
            if (w, h) != (full_w, full_h):
                degrades.append(f"resolution {full_w}x{full_h}->{w}x{h}")
            if speed_c != speed:
                degrades.append(f"{speed}->{speed_c}")
            steps_c = 4 if speed_c == "Fast" and speed == "Quality" else steps
            footprint = RESIDENT_GB + LORA_GB + act
            note = _note(act, act_budget, footprint, w, h, nref_c, cfg_c > 1.0, degrades)
            return dict(
                width=w, height=h, steps=steps_c, true_cfg=cfg_c, n_ref=nref_c,
                speed=speed_c, refused=False, note=note, degrades=degrades,
                act_gb=round(act, 1), act_budget_gb=round(act_budget, 1),
                need_gb=round(footprint, 1), budget_gb=round(RESIDENT_GB + LORA_GB + act_budget, 1),
            )

    # Hard floor: even the bottom rung's activation overflows free memory.
    w, h = fit_dims(base_w, base_h, RES_RUNGS[-1])
    act = _act_estimate(device, w, h, min_ref, False)
    footprint = RESIDENT_GB + LORA_GB + act
    note = (
        f"OOM-REFUSED: even {w}x{h} Fast needs ~{act:.0f} GB activation > free {act_budget:.0f} GB "
        f"(available {available_gb():.0f} - reserve {_reserve_gb():.0f}; weights {RESIDENT_GB:.0f} GB resident). "
        f"Free ~{act - act_budget:.0f} GB. Top: {top_consumers()}"
    )
    return dict(
        width=w, height=h, steps=4, true_cfg=1.0, n_ref=min_ref,
        speed="Fast", refused=True, note=note, degrades=["REFUSED"],
        act_gb=round(act, 1), act_budget_gb=round(act_budget, 1),
        need_gb=round(footprint, 1), budget_gb=round(RESIDENT_GB + LORA_GB + act_budget, 1),
    )


def _note(act, act_budget, footprint, w, h, n_ref, quality, degrades) -> str:
    base = (
        f"OOM preflight: activation ~{act:.0f} GB <= free {act_budget:.0f} GB "
        f"(weights {RESIDENT_GB + LORA_GB:.0f} GB resident; peak footprint ~{footprint:.0f} GB) "
        f"@ {w}x{h}{' Q' if quality else ' F'}, {n_ref + 1} image(s)."
    )
    if degrades:
        return base + " Degraded: " + "; ".join(degrades) + "."
    return base + " No degrade."


def penalize(device: str, height: int, width: int, n_ref: int, cfg_on: bool, failed_budget_gb: float) -> None:
    """Self-correcting calibration after an ACTUAL OOM (qwen-brain Q2.2).

    Bumps this config's cached estimate above the budget it just overflowed, so the next
    identical request degrades preemptively in the preflight instead of repeating the
    OOM + retry loop. Device-keyed (per record_peak).
    """
    key = (device, "act", _res_bucket(height, width), n_ref, cfg_on)
    _CALIB[key] = max(_CALIB.get(key, 0.0), failed_budget_gb * 1.2)


def step_down(width, height, n_ref, speed, true_cfg, steps, base_w, base_h, mode):
    """Next-more-aggressive config after an ACTUAL MPS OOM (reactive degrade).

    Mirrors the preflight ladder (REF-DROP first — the only strong lever; CFG then
    resolution are weak) so a mis-estimate converges to a fitting config instead of
    crashing. Each step strictly reduces memory and PRESERVES the user's step count
    (steps aren't a memory lever); only the Quality->Fast switch changes steps to the
    Lightning 4-step count. Returns a config dict, or None at the floor (-> clean refuse).
    This is the hard never-OOM guarantee.
    """
    # 1. (compose) drop a reference image — dominant lever, keep current resolution.
    if mode == "compose" and n_ref > 0:
        return dict(width=width, height=height, n_ref=n_ref - 1, speed=speed, true_cfg=true_cfg, steps=steps)
    # 2. (quality) drop CFG double-pass (weak).
    if true_cfg > 1.0:
        return dict(width=width, height=height, n_ref=n_ref, speed=speed, true_cfg=1.0, steps=steps)
    # 3. resolution down (weak lever) — last.
    cur = width * height
    idx = next((i for i, mp in enumerate(RES_RUNGS) if cur >= mp - 1), len(RES_RUNGS) - 1)
    if idx < len(RES_RUNGS) - 1:
        w, h = fit_dims(base_w, base_h, RES_RUNGS[idx + 1])
        if (w, h) != (width, height):
            return dict(width=w, height=h, n_ref=n_ref, speed=speed, true_cfg=true_cfg, steps=steps)
    # 4. force Quality->Fast (Lightning 4-step).
    if speed != "Fast":
        return dict(width=width, height=height, n_ref=n_ref, speed="Fast", true_cfg=1.0, steps=4)
    return None
