"""Cross-GPU parity check for the 2-step training smokes (plan section 14.5).

Compares two train_odace.py output dirs (same config/seed, run on different physical
GPUs) and verifies, within tolerance:
  - execution_mode / effective batch / seeds / config hash identical,
  - trainable layer manifest identical (module names, projections, parameter count;
    for scope=res16 additionally == 26,220,800 params over exactly 5 attn2 modules),
  - first- and last-step L_forget/L_retain/L_total relative diff <= --tol_loss
    (floating-point traces across two physical devices need not be bit-identical).

Writes models/odace/outputs/_smoke_parity_compare.json and exits 0 on PASS, 1 on FAIL.

Run:
  python models/odace/experiments/compare_smoke_runs.py \
      --run0 outputs/_smoke_ac_l1_r16_g0 --run1 outputs/_smoke_ac_l1_r16_g1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # models/odace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_smoke_runs")

RES16_PARAMS = 26_220_800
RES16_ATTN2 = 5


def _load(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    if not history:
        raise SystemExit(f"{run_dir}: empty history.json")
    return manifest, history


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run0", required=True)
    ap.add_argument("--run1", required=True)
    ap.add_argument("--tol_loss", type=float, default=1e-2)
    args = ap.parse_args()

    d0 = Path(args.run0) if Path(args.run0).is_absolute() else BASE / args.run0
    d1 = Path(args.run1) if Path(args.run1).is_absolute() else BASE / args.run1
    m0, h0 = _load(d0)
    m1, h1 = _load(d1)

    checks = {}
    for key in ("execution_mode", "config_sha256", "train_seed", "effective_batch_size",
                "target_mode", "target_lambda", "trainable_scope",
                "trainable_projections", "timestep_policy",
                "forget_prompts_sha256", "retain_prompts_sha256", "ood_prompts_sha256"):
        checks[f"{key}_equal"] = (m0.get(key) == m1.get(key))

    lm0, lm1 = m0.get("trainable_layer_manifest", {}), m1.get("trainable_layer_manifest", {})
    checks["module_names_equal"] = lm0.get("module_names") == lm1.get("module_names")
    checks["parameter_names_equal"] = (lm0.get("parameter_names")
                                       == lm1.get("parameter_names"))
    n0 = m0.get("trainable_parameter_count")
    n1 = m1.get("trainable_parameter_count")
    checks["trainable_count_equal"] = (n0 == n1)

    res16_checks = None
    if m0.get("trainable_scope") == "res16":
        mods = lm0.get("module_names") or []
        res16_checks = {
            "n_attn2_is_5": len(mods) == RES16_ATTN2,
            "params_are_26220800": n0 == RES16_PARAMS,
            "all_in_band": all(m.startswith(("down_blocks.2", "up_blocks.1"))
                               for m in mods),
        }
        checks.update({f"res16_{k}": v for k, v in res16_checks.items()})

    if len(h0) != len(h1):
        checks["history_length_equal"] = False
        loss_diffs = {}
    else:
        checks["history_length_equal"] = True
        loss_diffs = {}
        for tag, i in (("first", 0), ("last", len(h0) - 1)):
            for k in ("L_forget", "L_retain", "L_total"):
                loss_diffs[f"{tag}_{k}_rel_diff"] = _rel(h0[i][k], h1[i][k])
        checks["losses_within_tol"] = all(v <= args.tol_loss for v in loss_diffs.values())

    ok = all(checks.values())
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [str(d0), str(d1)],
        "gpu_uuids": [m0.get("physical_gpu_uuid"), m1.get("physical_gpu_uuid")],
        "gpu_uuids_distinct": (m0.get("physical_gpu_uuid") != m1.get("physical_gpu_uuid")
                               if m0.get("physical_gpu_uuid") and m1.get("physical_gpu_uuid")
                               else None),
        "trainable_parameter_count": [n0, n1],
        "checks": checks,
        "loss_rel_diffs": loss_diffs,
        "first_step_losses": [h0[0], h1[0]],
        "last_step_losses": [h0[-1], h1[-1]],
        "tol_loss": args.tol_loss,
        "PASS": bool(ok),
    }
    out_path = BASE / "outputs" / "_smoke_parity_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(json.dumps(out, indent=2))
    logger.info(f"SMOKE PARITY {'PASS' if ok else 'FAIL'} -> {out_path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
