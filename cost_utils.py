"""Environment-aware training-cost measurement for ODACE runs.

Records the ACTUAL compute cost of a training run on whatever machine runs it, so that a
fresh `git clone` + identical re-train re-measures cost *for that environment* (GPU model,
wall time, VRAM) instead of trusting committed reference numbers. Cost is reported in
hardware-neutral units (GPU-hours, trainable params, peak VRAM) plus the GPU name so that
cross-machine numbers stay interpretable. No USD estimate is produced.

Usage in a trainer:

    from cost_utils import CostMeter

    with CostMeter("odace", output_dir, steps=cfg.num_steps) as meter:
        meter.set_trainable_params(trainable_module_or_param_iter)
        ... training loop ...
    # on __exit__ writes <output_dir>/train_cost.json with this machine's measurements
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


def _gpu_name() -> str:
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def _gpu_count() -> int:
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def count_trainable_params(obj) -> int:
    """Sum of requires_grad params. Accepts an nn.Module or any iterable of params."""
    params = obj.parameters() if hasattr(obj, "parameters") else obj
    return int(sum(p.numel() for p in params if getattr(p, "requires_grad", False)))


class CostMeter:
    """Context manager that times a training run and records env-aware cost to JSON.

    Wall time is measured on the host running this code, so GPU-hours reflect *this* environment.
    peak_vram_gb uses torch's per-process CUDA peak allocator counter (reset on enter).
    """

    def __init__(self, model: str, output_dir, steps: int | None = None):
        self.model = model
        self.output_dir = Path(output_dir)
        self.steps = steps
        self.trainable_params = None
        self._t0 = None

    def set_trainable_params(self, obj) -> None:
        self.trainable_params = count_trainable_params(obj)

    def __enter__(self):
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        wall = time.perf_counter() - (self._t0 or time.perf_counter())
        gc = _gpu_count()
        peak_vram = 0.0
        if _HAS_TORCH and torch.cuda.is_available():
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
        rec = {
            "model": self.model,
            "training_free": False,
            "gpu": _gpu_name(),
            "gpu_count": gc,
            "trainable_params_M": round(self.trainable_params / 1e6, 2)
            if self.trainable_params is not None else None,
            "steps": self.steps,
            "wall_seconds": round(wall, 1),
            "gpu_hours": round(wall * max(gc, 1) / 3600.0, 3),
            "peak_vram_gb": peak_vram,
            "torch": torch.__version__ if _HAS_TORCH else None,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "status": "ok" if exc_type is None else f"interrupted:{exc_type.__name__}",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train_cost.json").write_text(json.dumps(rec, indent=2))
        return False  # never suppress exceptions
