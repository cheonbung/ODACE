"""P1 regression gate: frozen legacy trainer vs refactored trainer (legacy_exact mode).

Runs BOTH implementations for --steps (1-5) optimizer steps on the SAME legacy config,
seed and prompt data, sequentially in one process, and compares per step:
  - the sampled trajectory index t_enc_idx and the timestep fed to the UNet (must be int-equal),
  - the trajectory latent z and the (stop-gradient) target tensor (max abs diff <= --tol_tensor),
  - L_forget / L_retain / L_total (rel diff <= --tol_loss),
and after the last step the trainable parameters (max abs diff + SHA-256 of the
concatenated fp32 bytes). Also asserts the trainable parameter count matches between
implementations (and equals 43,962,560 for the full q/k/v/out scope on SD1.4).

Prompt draws are made here from the GLOBAL `random` module in the legacy order
(choice(forget), choice(retain), then the trainer's randint/randn); both trainers reseed
the global RNG to the same seed at construction, so the two runs consume identical
streams.

GPU backward kernels are NOT run-to-run deterministic (atomics), and Adam normalizes the
update so a ~1e-8 gradient wiggle can become a ~lr-sized weight wiggle. To separate that
hardware noise from implementation drift, the legacy snapshot is run TWICE (A/A control)
with the identical seed: the legacy-vs-refactor weight/loss differences must not exceed
max(tolerance, 3 x the A/A difference). Per-step targets are compared strictly (they are
produced by the forward-only teacher path, which IS deterministic).

Writes (plan P1 deliverables):
  models/odace/outputs/_regression_legacy/manifest.json
  models/odace/outputs/_regression_refactor/manifest.json
  models/odace/outputs/_regression_compare.json     <- per-step diffs + PASS/FAIL

Run (WSL conda lsse, ~2-3 min on one GPU):
  CUDA_VISIBLE_DEVICES=0 python models/odace/experiments/regression_compare.py \
      --config configs/nudity_odace_benign_n1.yaml --steps 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

BASE = Path(__file__).resolve().parents[1]          # models/odace
sys.path.insert(0, str(BASE))

from core.dataset import DACEDataset                          # noqa: E402
from core.config_schema import load_config                    # noqa: E402
from core.trainer import ODACETrainer                         # noqa: E402
from core.legacy_trainer_snapshot import LegacyODACETrainer   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("regression_compare")

FULL_QKVO_SD14 = 43_962_560   # 16 attn2 x q,k,v,out(+bias) on SD1.4


def _build_dataset(cfg) -> DACEDataset:
    dataset = DACEDataset.from_files(
        forget_file=str(BASE / cfg.forget_prompts_file),
        retain_file=str(BASE / cfg.retain_prompts_file),
        target_concept=cfg.target_concept, seed=cfg.seed)
    if cfg.ood_aug_file:
        ood_path = Path(cfg.ood_aug_file)
        if not ood_path.is_absolute():
            ood_path = BASE / cfg.ood_aug_file
        ood = [ln.strip() for ln in ood_path.read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.startswith("#")]
        dataset.forget_prompts = dataset.forget_prompts + ood
    return dataset


def _trainable_state(unet):
    parts, names = [], []
    for name, p in sorted(unet.named_parameters()):
        if p.requires_grad:
            parts.append(p.detach().float().cpu().reshape(-1))
            names.append(name)
    flat = torch.cat(parts)
    sha = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
    return flat, sha, names


def _run(trainer, dataset, steps: int):
    """Drive `steps` optimizer steps in the legacy train() order, capturing debug info."""
    records = []
    for step in range(1, steps + 1):
        bf = [random.choice(dataset.forget_prompts)]
        br = [random.choice(dataset.retain_prompts)]
        s = trainer._train_step(bf, br)
        dbg = trainer.last_debug
        records.append({"step": step,
                        **{k: float(v) for k, v in s.items() if k != "step"},
                        "t": dbg["t"], "t_enc_idx": dbg["t_enc_idx"],
                        "forget_prompt": dbg["forget_prompt"],
                        "retain_prompt": dbg["retain_prompt"],
                        "_target": dbg["target"], "_z": dbg["z"]})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nudity_odace_benign_n1.yaml")
    ap.add_argument("--steps", type=int, default=2, choices=range(1, 6))
    ap.add_argument("--device", default=None)
    ap.add_argument("--tol_tensor", type=float, default=1e-4)
    ap.add_argument("--tol_loss", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = BASE / args.config
    cfg = load_config(cfg_path)
    if cfg.execution_mode != "legacy_exact":
        raise SystemExit("regression compare requires a LEGACY config (legacy_exact)")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    dataset = _build_dataset(cfg)
    logger.info(f"config={args.config} steps={args.steps} device={device} "
                f"forget={len(dataset.forget_prompts)} retain={len(dataset.retain_prompts)}")

    def build_legacy():
        return LegacyODACETrainer(
            sd_model_id=cfg.sd_model_id, device=device,
            learning_rate=cfg.learning_rate, alpha=cfg.alpha, beta=cfg.beta,
            eta=float(raw.get("eta", 1.0)), ddim_steps=cfg.ddim_steps,
            sample_guidance=cfg.sample_guidance,
            xattn_full=bool(raw.get("xattn_full", False)),
            erase_mode=str(raw.get("erase_mode", "negguide")),
            benign_prompt=str(raw.get("benign_prompt",
                                      "a fully clothed person, photograph")),
            benign_neg_lambda=float(raw.get("benign_neg_lambda", 1.0)),
            seed=cfg.seed)

    # ---- run 1: frozen legacy snapshot ------------------------------------------
    legacy = build_legacy()
    legacy.capture_debug = True
    n_train_legacy = sum(p.numel() for p in legacy.unet.parameters() if p.requires_grad)
    legacy_records = _run(legacy, dataset, args.steps)
    legacy_flat, legacy_sha, legacy_names = _trainable_state(legacy.unet)
    del legacy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- run 1b: legacy snapshot AGAIN -- A/A nondeterminism floor ----------------
    legacy2 = build_legacy()
    legacy2.capture_debug = True
    legacy2_records = _run(legacy2, dataset, args.steps)
    legacy2_flat, legacy2_sha, _ = _trainable_state(legacy2.unet)
    del legacy2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- run 2: refactored trainer, legacy_exact --------------------------------
    new = ODACETrainer(cfg, device)
    new.capture_debug = True
    n_train_new = new.selection.trainable_parameter_count
    new_records = _run(new, dataset, args.steps)
    new_flat, new_sha, new_names = _trainable_state(new.unet)

    # ---- A/A nondeterminism floor --------------------------------------------------
    aa_w_diff = float((legacy_flat - legacy2_flat).abs().max())
    aa_loss_rel = max(
        abs(a[k] - b[k]) / max(abs(a[k]), 1e-12)
        for a, b in zip(legacy_records, legacy2_records)
        for k in ("L_forget", "L_retain", "L_total"))
    eff_w_tol = max(args.tol_tensor, 3.0 * aa_w_diff)
    eff_loss_tol = max(args.tol_loss, 3.0 * aa_loss_rel)

    # ---- compare -----------------------------------------------------------------
    steps_cmp, ok = [], True
    for a, b in zip(legacy_records, new_records):
        target_diff = float((a["_target"] - b["_target"]).abs().max())
        z_diff = float((a["_z"] - b["_z"]).abs().max())
        loss_rel = max(
            abs(a[k] - b[k]) / max(abs(a[k]), 1e-12)
            for k in ("L_forget", "L_retain", "L_total"))
        row = {
            "step": a["step"],
            "prompts_equal": (a["forget_prompt"] == b["forget_prompt"]
                              and a["retain_prompt"] == b["retain_prompt"]),
            "t_enc_idx": [a["t_enc_idx"], b["t_enc_idx"]],
            "t": [a["t"], b["t"]],
            "z_max_abs_diff": z_diff,
            "target_max_abs_diff": target_diff,
            "L_forget": [a["L_forget"], b["L_forget"]],
            "L_retain": [a["L_retain"], b["L_retain"]],
            "L_total": [a["L_total"], b["L_total"]],
            "loss_max_rel_diff": loss_rel,
        }
        row["pass"] = (row["prompts_equal"]
                       and a["t_enc_idx"] == b["t_enc_idx"] and a["t"] == b["t"]
                       and z_diff <= args.tol_tensor
                       and target_diff <= args.tol_tensor
                       and loss_rel <= eff_loss_tol)
        ok &= row["pass"]
        steps_cmp.append(row)

    w_diff = float((legacy_flat - new_flat).abs().max()) \
        if legacy_flat.numel() == new_flat.numel() else float("inf")
    counts_ok = (n_train_legacy == n_train_new)
    full_scope_ok = (n_train_new == FULL_QKVO_SD14) if bool(raw.get("xattn_full")) else None
    names_ok = (legacy_names == new_names)
    ok &= counts_ok and names_ok and (w_diff <= eff_w_tol)
    if full_scope_ok is False:
        ok = False

    now = datetime.now(timezone.utc).isoformat()
    common = {"created_utc": now, "config": str(cfg_path),
              "config_sha256": cfg.config_sha256, "train_seed": cfg.seed,
              "num_steps": args.steps, "device": str(device),
              "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None}

    def strip(recs):
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in recs]

    out_root = BASE / "outputs"
    for sub, recs, sha, n in (("_regression_legacy", legacy_records, legacy_sha, n_train_legacy),
                              ("_regression_refactor", new_records, new_sha, n_train_new)):
        d = out_root / sub
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({**common,
                       "implementation": ("legacy_snapshot" if "legacy" in sub
                                          else "refactored_legacy_exact"),
                       "trainable_parameter_count": n,
                       "final_trainable_sha256": sha,
                       "steps": strip(recs)}, f, indent=2)

    compare = {**common,
               "steps": steps_cmp,
               "trainable_parameter_count": [n_train_legacy, n_train_new],
               "trainable_count_equal": counts_ok,
               "trainable_names_equal": names_ok,
               "full_scope_expected_43962560": full_scope_ok,
               "final_weights_max_abs_diff": w_diff,
               "final_weights_sha256": [legacy_sha, new_sha],
               "final_weights_sha_equal": legacy_sha == new_sha,
               "aa_control": {
                   "note": "legacy snapshot run twice, identical seed: GPU backward "
                           "nondeterminism floor; refactor diffs gated at 3x this",
                   "aa_final_weights_max_abs_diff": aa_w_diff,
                   "aa_loss_max_rel_diff": aa_loss_rel,
                   "aa_final_weights_sha256": [legacy_sha, legacy2_sha]},
               "tolerances": {"tensor": args.tol_tensor, "loss_rel": args.tol_loss,
                              "effective_weight_tol": eff_w_tol,
                              "effective_loss_tol": eff_loss_tol},
               "PASS": bool(ok)}
    with open(out_root / "_regression_compare.json", "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2)

    logger.info(json.dumps({k: v for k, v in compare.items() if k != "steps"}, indent=2))
    for row in steps_cmp:
        logger.info(f"step {row['step']}: pass={row['pass']} "
                    f"target_dmax={row['target_max_abs_diff']:.3e} "
                    f"loss_drel={row['loss_max_rel_diff']:.3e} t={row['t']}")
    logger.info(f"REGRESSION {'PASS' if ok else 'FAIL'} -> {out_root/'_regression_compare.json'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
