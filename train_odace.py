"""ODACE training entry. Run from the repository root:

    python train_odace.py --config configs/nudity_odace_benign_n1.yaml
    python train_odace.py --config configs/paper/nudity_ac_l1_r16.yaml \
        --num_steps 2 --output_dir outputs/_smoke_ac_l1_r16_g0

Configs are validated by core/config_schema.py (unknown keys are errors; legacy configs
migrate to execution_mode=legacy_exact with the migration recorded). Every run writes,
BEFORE training starts:
  <output_dir>/resolved_config.yaml  -- the fully-resolved config incl. migration notes
  <output_dir>/manifest.json         -- schema odace-train-v1: git commit + dirty flag,
                                        code/config/prompt SHA-256, base model id and
                                        cached revision (best effort), seeds/RNG record,
                                        trainable layer manifest + parameter counts,
                                        GPU/driver/software versions
and after training updates manifest.json with forward/backward call counters and writes
history.json (legacy format) + train_cost.json (CostMeter).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

BASE = Path(__file__).resolve().parent
REPO = BASE  # flattened release layout: repo root == this file's directory
sys.path.insert(0, str(BASE))

from core.dataset import DACEDataset                    # noqa: E402
from core.config_schema import load_config              # noqa: E402
from core.trainer import ODACETrainer                   # noqa: E402
from cost_utils import CostMeter                        # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_odace")

# files whose content defines "the training code" for provenance hashing
_CODE_FILES = (
    "train_odace.py", "core/trainer.py", "core/targets.py", "core/config_schema.py",
    "core/dataset.py", "methods/layer_selection.py", "methods/unet_edit.py",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_sha256() -> str:
    h = hashlib.sha256()
    for rel in _CODE_FILES:
        p = BASE / rel
        h.update(rel.encode())
        h.update(_sha256_file(p).encode())
    return h.hexdigest()


def _git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    status = run("status", "--short")
    return {
        "git_commit": run("rev-parse", "HEAD") or None,
        "dirty_worktree": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest() if status else None,
    }


def _gpu_info() -> dict:
    import os
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    info = {"gpu_name": None, "gpu_memory_total_mib": None, "physical_gpu_index": None,
            "physical_gpu_uuid": None, "driver_version": None,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "cuda_visible_devices": cvd, "logical_cuda_device": None}
    if torch.cuda.is_available():
        info["logical_cuda_device"] = torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            # nvidia-smi IGNORES CUDA_VISIBLE_DEVICES, so map logical cuda:0 back to the
            # physical row: numeric CVD -> that physical index; UUID CVD -> uuid match;
            # unset -> the logical index is the physical index.
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,memory.total,driver_version",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            rows = [[c.strip() for c in r.split(",")]
                    for r in out.stdout.strip().splitlines() if r.strip()]
            first_visible = (cvd.split(",")[0].strip() if cvd else None)
            row = None
            if first_visible is None:
                row = rows[torch.cuda.current_device()]
            elif first_visible.isdigit():
                row = next((r for r in rows if int(r[0]) == int(first_visible)), None)
            else:
                row = next((r for r in rows if r[1].startswith(first_visible)), None)
            if row:
                info["physical_gpu_index"] = int(row[0])
                info["physical_gpu_uuid"] = row[1]
                info["gpu_memory_total_mib"] = int(row[2].split()[0])
                info["driver_version"] = row[3]
        except Exception:  # noqa: BLE001
            pass
    return info


def _base_model_revision(model_id: str) -> dict:
    """Best-effort: resolve the locally-cached snapshot commit for provenance."""
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id:
                revs = sorted(repo.revisions, key=lambda r: r.last_modified or 0)
                if revs:
                    return {"base_model_revision": revs[-1].commit_hash}
    except Exception:  # noqa: BLE001
        pass
    return {"base_model_revision": None}


def build_manifest(cfg, dataset_files: dict, trainer=None) -> dict:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for mod in ("diffusers", "transformers"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = None
    m = {
        "schema_version": "odace-train-v1",
        "experiment_id": cfg.experiment_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **_git_info(),
        "code_sha256": _code_sha256(),
        "base_model_id": cfg.sd_model_id,
        **_base_model_revision(cfg.sd_model_id),
        "config_sha256": cfg.config_sha256,
        "config_source_path": cfg.source_path,
        "execution_mode": cfg.execution_mode,
        "target_mode": cfg.target_mode,
        "target_lambda": cfg.target_lambda,
        "eta": cfg.eta,
        "anchor_policy": cfg.anchor_policy,
        "anchor_prompt": cfg.anchor_prompt,
        "trainable_scope": cfg.trainable_scope,
        "trainable_projections": list(cfg.trainable_projections),
        "timestep_policy": cfg.timestep_policy,
        "trajectory_index_min": cfg.trajectory_index_min,
        "trajectory_index_max": cfg.trajectory_index_max,
        "num_optimizer_steps": cfg.num_optimizer_steps,
        "batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "effective_batch_size": cfg.effective_batch_size,
        "train_seed": cfg.seed,
        "migration_notes": cfg.migration_notes,
        "ignored_legacy_keys": cfg.ignored_legacy_keys,
        "software_versions": versions,
        **_gpu_info(),
    }
    for label, path in dataset_files.items():
        m[f"{label}_sha256"] = _sha256_file(Path(path)) if path else None
        m[f"{label}_path"] = str(path) if path else None
    if trainer is not None:
        sel = trainer.selection.to_manifest()
        m["rng_record"] = trainer.rng_record
        m["trainable_parameter_count"] = sel["trainable_parameter_count"]
        m["total_cross_attention_parameter_count"] = \
            sel["total_cross_attention_parameter_count"]
        m["trainable_layer_manifest"] = {
            "scope": sel["scope"], "projections": sel["projections"],
            "module_names": sel["module_names"],
            "n_parameter_tensors": len(sel["parameter_names"]),
            "parameter_names": sel["parameter_names"],
            "architecture_fingerprint": sel["architecture_fingerprint"],
        }
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML path relative to the repository root")
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = BASE / cfg_path
    overrides = {k: getattr(args, k) for k in
                 ("num_steps", "alpha", "beta", "batch_size", "output_dir", "seed")}
    cfg = load_config(cfg_path, overrides=overrides)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} exp={cfg.experiment_name} mode={cfg.execution_mode}")
    if cfg.migration_notes:
        for note in cfg.migration_notes:
            logger.warning(f"[config-migration] {note}")

    forget_file = BASE / cfg.forget_prompts_file
    retain_file = BASE / cfg.retain_prompts_file
    dataset = DACEDataset.from_files(
        forget_file=str(forget_file), retain_file=str(retain_file),
        target_concept=cfg.target_concept, seed=cfg.seed)
    logger.info(repr(dataset))

    ood_path = None
    if cfg.ood_aug_file:
        ood_path = Path(cfg.ood_aug_file)
        if not ood_path.is_absolute():
            ood_path = BASE / cfg.ood_aug_file
        if not ood_path.exists():
            logger.warning(f"[OOD-aug] file not found: {ood_path} -- skipping OOD augmentation")
            ood_path = None
        else:
            ood = [ln.strip() for ln in ood_path.read_text(encoding="utf-8").splitlines()
                   if ln.strip() and not ln.startswith("#")]
            dataset.forget_prompts = dataset.forget_prompts + ood
            logger.info(f"[OOD-aug] +{len(ood)} OOD prompts -> forget "
                        f"(total {len(dataset.forget_prompts)})")

    trainer = ODACETrainer(cfg, device)

    out_dir = Path(cfg.output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = {"forget_prompts": forget_file, "retain_prompts": retain_file,
                     "ood_prompts": ood_path}
    manifest = build_manifest(cfg, dataset_files, trainer)
    with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_manifest_dict(), f, sort_keys=False, allow_unicode=True)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"manifest written -> {out_dir}/manifest.json (pre-training)")

    with CostMeter(cfg.experiment_name, str(out_dir),
                   steps=cfg.num_optimizer_steps) as meter:
        history = trainer.train(dataset)
        meter.set_trainable_params(trainer.unet)
    trainer.save(str(out_dir / "final"))
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    manifest["counters"] = trainer.counters
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["final_losses"] = history[-1] if history else None
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"DONE -> {out_dir}/final ; history.json ({len(history)} steps); "
                f"manifest.json updated with counters")


if __name__ == "__main__":
    main()
